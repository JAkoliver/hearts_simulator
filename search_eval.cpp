// SearchEval: paired-deal evaluation of the decision-time search player,
// plus a --selftest mode validating the search machinery.
//
// Usage:
//   SearchEval --search-model <pt> --selftest [--cuda]
//   SearchEval --search-model <pt> --opponent-model <pt> --deals 300 --k 32
//              --seed 42 [--uniform-sampling] [--pass-search] [--cuda]
//              --out results.csv
//
// --cuda moves the search player's inference to the GPU (the opponent's
// raw batch-1 policy always stays on CPU, where it is faster).

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "HeartsEnv.hpp"
#include "SearchPlayer.hpp"
#include "TreeSearchPlayer.hpp"

// ---------------------------------------------------------------------------
// Match-to-100 bridge (docs/ROADMAP.md "Queued: match-aware search", step 1)
// ---------------------------------------------------------------------------

// Raw policy over a 556-dim match trace: engine obs (550) + the 6 match
// context dims, mirroring hearts_match_env.match_ctx_row exactly:
// [self,left,across,right scores]/100, deals/20, leader-distance-to-100/100.
class MatchRawPolicy {
public:
    MatchRawPolicy(torch::jit::script::Module model) : model_(std::move(model)) {}

    int ChooseAction(const HeartsEnv& env, const std::array<double, 4>& totals,
                     int deals_played) {
        auto obs = env.Observe();  // 550 floats
        int me = env.GetCurrentPlayer();
        torch::Tensor o = torch::zeros({1, 556}, torch::kFloat32);
        float* op = o.data_ptr<float>();
        std::copy(obs.begin(), obs.end(), op);
        double mx = *std::max_element(totals.begin(), totals.end());
        for (int i = 0; i < 4; ++i) op[550 + i] = (float)(totals[(me + i) % 4] / 100.0);
        op[554] = (float)(deals_played / 20.0);
        op[555] = (float)((100.0 - mx) / 100.0);

        auto legal_raw = env.GetLegalActions();
        torch::Tensor m = torch::zeros({1, 52}, torch::kBool);
        bool* mp = m.data_ptr<bool>();
        for (int i = 0; i < 13; ++i) {
            if (legal_raw[i] != -1) mp[legal_raw[i]] = true;
        }
        torch::NoGradGuard g;
        auto out = model_.forward({o, m}).toTuple();
        return out->elements()[0].toTensor().argmax(1).item<int>();
    }

private:
    torch::jit::script::Module model_;
};

// 1 = winner (lowest total); ties share the mean of their ranks.
static std::array<double, 4> Placements(const std::array<double, 4>& totals) {
    std::array<int, 4> order = {0, 1, 2, 3};
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return totals[a] < totals[b]; });
    std::array<double, 4> ranks{};
    int i = 0;
    while (i < 4) {
        int j = i;
        while (j + 1 < 4 && totals[order[j + 1]] == totals[order[i]]) ++j;
        double shared = (i + j) / 2.0 + 1.0;
        for (int t = i; t <= j; ++t) ranks[order[t]] = shared;
        i = j + 1;
    }
    return ranks;
}

static int RandomLegal(const HeartsEnv& env, std::mt19937& rng) {
    auto lr = env.GetLegalActions();
    std::vector<int> legal;
    for (int i = 0; i < 13; ++i) {
        if (lr[i] != -1) legal.push_back(lr[i]);
    }
    return legal[rng() % legal.size()];
}

// ---------------------------------------------------------------------------
// Selftest
// ---------------------------------------------------------------------------

static bool TestCloneFidelity() {
    HeartsEnv e(11, true);
    e.Reset();
    std::mt19937 warm(5);
    for (int i = 0; i < 20; ++i) e.Step(RandomLegal(e, warm));

    HeartsEnv c = e.Clone();
    std::mt19937 ra(9), rb(9);
    bool done = false;
    while (!done) {
        if (e.GetCurrentPlayer() != c.GetCurrentPlayer()) return false;
        if (e.GetLegalActions() != c.GetLegalActions()) return false;
        if (e.Observe() != c.Observe()) return false;
        int a1 = RandomLegal(e, ra);
        int a2 = RandomLegal(c, rb);
        if (a1 != a2) return false;
        done = e.Step(a1).done;
        if (c.Step(a2).done != done) return false;
    }
    return e.GetRoundScores() == c.GetRoundScores();
}

static bool TestDeterminizations(torch::jit::script::Module& model, int obs_dim,
                                 torch::Device device) {
    SearchPlayer::Config cfg;
    cfg.determinizations = 8;
    cfg.seed = 99;
    cfg.device = device;
    SearchPlayer sp(model, obs_dim, cfg);
    std::mt19937 rr(17);
    HeartsEnv env(23, true);
    int checked = 0, playthroughs = 0;

    for (int round = 0; round < 30; ++round) {
        env.Reset();
        bool done = false;
        while (!done) {
            if (!env.IsPassing() && rr() % 3 == 0) {
                sp.BuildContext(env);
                int me = env.GetCurrentPlayer();
                const auto& played_by = env.GetPlayedBy();
                const auto& voids = env.GetVoidTracker();

                auto hands = sp.SampleHands(env);

                // Sizes
                for (int p = 0; p < 4; ++p) {
                    if ((int)hands[p].size() != env.GetHandSize(p)) return false;
                }
                // Partition: exactly the unplayed cards, no duplicates
                std::set<int> all;
                for (int p = 0; p < 4; ++p) {
                    for (int id : hands[p]) {
                        if (!all.insert(id).second) return false;
                    }
                }
                for (int cd = 0; cd < 52; ++cd) {
                    bool played = played_by[0][cd] || played_by[1][cd] || played_by[2][cd] || played_by[3][cd];
                    if (played == (all.count(cd) > 0)) return false;
                }
                // Voids never violated
                for (int k = 1; k < 4; ++k) {
                    int seat = (me + k) % 4;
                    for (int id : hands[seat]) {
                        if (voids[seat * 4 + (id / 13)]) return false;
                    }
                }
                // Passed-card pinning
                if (env.GetPassDirection() != 3) {
                    int off = (env.GetPassDirection() == 0) ? 1 : (env.GetPassDirection() == 1) ? 3 : 2;
                    int receiver = (me + off) % 4;
                    for (int a : env.GetPassPicks(me)) {
                        bool played = played_by[0][a] || played_by[1][a] || played_by[2][a] || played_by[3][a];
                        if (!played) {
                            bool held = std::find(hands[receiver].begin(), hands[receiver].end(), a)
                                        != hands[receiver].end();
                            if (!held) return false;
                        }
                    }
                }
                checked++;

                // SetHands + full random playthrough stays rule-consistent
                if (playthroughs < 30) {
                    HeartsEnv sim = env.Clone();
                    sim.SetHands(hands);
                    std::mt19937 pr(rr());
                    bool sdone = false;
                    while (!sdone) sdone = sim.Step(RandomLegal(sim, pr)).done;
                    auto sc = sim.GetRoundScores();
                    int total = sc[0] + sc[1] + sc[2] + sc[3];
                    if (total != 26 && total != 78) return false;
                    playthroughs++;
                }
            }
            done = env.Step(RandomLegal(env, rr)).done;
        }
    }
    std::cerr << "  (validated " << checked << " determinizations, "
              << playthroughs << " playthroughs)\n";
    return checked >= 200;
}

static bool TestTree(torch::jit::script::Module& model, int obs_dim, torch::Device device) {
    TreeSearchPlayer::Config tcfg;
    tcfg.iterations = 48;
    tcfg.leaf_batch = 8;
    tcfg.seed = 7;
    tcfg.pass_cfg.device = device;
    TreeSearchPlayer tp(model, obs_dim, tcfg);
    std::mt19937 rr(21);
    HeartsEnv env(37, true);
    int checked = 0;

    for (int round = 0; round < 8; ++round) {
        env.Reset();
        bool done = false;
        while (!done) {
            int a;
            if (!env.IsPassing() && rr() % 4 == 0) {
                a = tp.ChooseAction(env);
                auto lr = env.GetLegalActions();
                bool legal = false;
                for (int i = 0; i < 13; ++i) {
                    if (lr[i] == a) legal = true;
                }
                if (!legal) return false;
                float psum = 0.0f;
                for (int c = 0; c < 52; ++c) psum += tp.LastPolicy()[c];
                if (std::abs(psum - 1.0f) > 1e-3f) return false;
                checked++;
            } else {
                a = RandomLegal(env, rr);
            }
            done = env.Step(a).done;
        }
    }
    std::cerr << "  (tree decisions checked: " << checked << ")\n";
    return checked >= 30;
}

static bool TestBatchEquivalence(torch::jit::script::Module& model, int obs_dim,
                                 torch::Device device) {
    HeartsEnv env(31, true);
    env.Reset();
    std::mt19937 rr(3);
    std::vector<std::array<float, 550>> states;
    std::vector<std::array<int, 13>> legals;
    bool done = false;
    while (!done && states.size() < 8) {
        if (!env.IsPassing()) {
            states.push_back(env.Observe());
            legals.push_back(env.GetLegalActions());
        }
        done = env.Step(RandomLegal(env, rr)).done;
    }

    int n = static_cast<int>(states.size());
    torch::Tensor ob = torch::empty({n, obs_dim}, torch::kFloat32);
    torch::Tensor mb = torch::zeros({n, 52}, torch::kBool);
    for (int i = 0; i < n; ++i) {
        std::memcpy(ob.data_ptr<float>() + (size_t)i * obs_dim, states[i].data(), obs_dim * sizeof(float));
        for (int j = 0; j < 13; ++j) {
            if (legals[i][j] != -1) mb.data_ptr<bool>()[i * 52 + legals[i][j]] = true;
        }
    }
    torch::NoGradGuard g;
    // GPU kernels may reorder reductions between batch sizes; allow a slightly
    // looser tolerance there than on CPU.
    double tol = device.is_cuda() ? 1e-3 : 1e-4;
    torch::Tensor obd = ob.to(device), mbd = mb.to(device);
    torch::Tensor batched = model.forward({obd, mbd}).toTuple()->elements()[0].toTensor();
    for (int i = 0; i < n; ++i) {
        torch::Tensor single = model.forward({obd.slice(0, i, i + 1), mbd.slice(0, i, i + 1)})
                                   .toTuple()->elements()[0].toTensor();
        torch::Tensor mask_row = mbd.slice(0, i, i + 1);
        double diff = (batched.slice(0, i, i + 1).masked_select(mask_row)
                       - single.masked_select(mask_row)).abs().max().item<double>();
        if (diff > tol) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    // Engine work is plain C++ on this thread; without this, libtorch's
    // intra-op pool fans tiny CPU ops across every core - harmless on a
    // 16-thread desktop, but on a 224-thread cloud box the spin-wait
    // coordination cost STALLED the process entirely (measured 2026-07-17:
    // 0 deals in 25 min vs 0.95 s/deal with the pool pinned).
    torch::set_num_threads(1);
    std::string search_path, opp_path, belief_path, match_path, probe_log,
        b_search_path, equity_path, behave_totals, out_path = "search_eval_results.csv";
    int behave_deals = 200;
    int deals = 300, k = 32, rollout_tricks = -1, matches = 200, probe_every = 5,
        k_endgame = 0;
    int tree_iterations = 400;
    float tree_c_puct = 1.5f;
    unsigned int seed = 42;
    bool uniform = false, selftest = false, pass_search = false, use_cuda = false;
    bool oracle_leaves = false, use_tree = false, use_bf16 = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--search-model") search_path = next();
        else if (a == "--opponent-model") opp_path = next();
        else if (a == "--belief-model") belief_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--match-model") match_path = next();
        else if (a == "--matches") matches = std::stoi(next());
        else if (a == "--probe-log") probe_log = next();
        else if (a == "--probe-every") probe_every = std::stoi(next());
        else if (a == "--k-endgame") k_endgame = std::stoi(next());
        else if (a == "--b-search-model") b_search_path = next();
        else if (a == "--equity-model") equity_path = next();
        else if (a == "--behave-totals") behave_totals = next();
        else if (a == "--behave-deals") behave_deals = std::stoi(next());
        else if (a == "--deals") deals = std::stoi(next());
        else if (a == "--k") k = std::stoi(next());
        else if (a == "--seed") seed = static_cast<unsigned int>(std::stoul(next()));
        else if (a == "--rollout-tricks") rollout_tricks = std::stoi(next());
        else if (a == "--oracle-leaves") oracle_leaves = true;
        else if (a == "--tree") use_tree = true;
        else if (a == "--iterations") tree_iterations = std::stoi(next());
        else if (a == "--c-puct") tree_c_puct = std::stof(next());
        else if (a == "--uniform-sampling") uniform = true;
        else if (a == "--pass-search") pass_search = true;
        else if (a == "--selftest") selftest = true;
        else if (a == "--cuda") use_cuda = true;
        else if (a == "--bf16") use_bf16 = true;
        else { std::cerr << "Unknown arg: " << a << "\n"; return 2; }
    }
    if (search_path.empty()) {
        std::cerr << "--search-model is required\n";
        return 2;
    }
    if (use_cuda && !torch::cuda::is_available()) {
        std::cerr << "--cuda requested but CUDA is not available in this libtorch build\n";
        return 1;
    }
    torch::Device device = use_cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    torch::jit::script::Module search_model;
    try {
        search_model = torch::jit::load(search_path);
        search_model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Failed to load search model: " << e.what() << "\n";
        return 1;
    }
    int sdim = ProbeObsDim(search_model);
    if (sdim == 0) {
        std::cerr << "Search model rejected all known observation widths\n";
        return 1;
    }

    if (selftest) {
        bool ok = true;
        std::cerr << "Clone fidelity... ";
        bool t = TestCloneFidelity();
        std::cerr << (t ? "PASS" : "FAIL") << "\n";
        ok &= t;
        std::cerr << "Determinization validity... ";
        t = TestDeterminizations(search_model, sdim, device);
        std::cerr << (t ? "PASS" : "FAIL") << "\n";
        ok &= t;
        std::cerr << "Batched-vs-single inference... ";
        t = TestBatchEquivalence(search_model, sdim, device);
        std::cerr << (t ? "PASS" : "FAIL") << "\n";
        ok &= t;
        std::cerr << "Tree search consistency... ";
        t = TestTree(search_model, sdim, device);
        std::cerr << (t ? "PASS" : "FAIL") << "\n";
        ok &= t;
        std::cerr << (ok ? "SELFTEST PASS" : "SELFTEST FAIL") << "\n";
        return ok ? 0 : 1;
    }

    if (opp_path.empty()) {
        std::cerr << "--opponent-model is required for a match\n";
        return 2;
    }
    torch::jit::script::Module opp_model;
    try {
        opp_model = torch::jit::load(opp_path);
        opp_model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Failed to load opponent model: " << e.what() << "\n";
        return 1;
    }
    int odim = ProbeObsDim(opp_model);
    if (odim == 0) {
        std::cerr << "Opponent model rejected all known observation widths\n";
        return 1;
    }

    SearchPlayer::Config cfg;
    cfg.determinizations = k;
    cfg.belief_weighted = !uniform;
    cfg.seed = seed + 1000;
    cfg.pass_search = pass_search;
    cfg.device = device;
    cfg.bf16 = use_bf16;
    cfg.rollout_tricks = rollout_tricks;
    cfg.oracle_leaves = oracle_leaves;
    cfg.probe_log = probe_log;
    cfg.probe_every = probe_every;
    cfg.k_endgame = k_endgame;
    if (!equity_path.empty()) {
        // Arm A becomes MATCH-AWARE: 556-dim search trace + equity leaf scoring
        try {
            cfg.equity_model = std::make_shared<torch::jit::script::Module>(
                torch::jit::load(equity_path));
            cfg.equity_model->eval();
        } catch (const c10::Error& e) {
            std::cerr << "Failed to load equity model: " << e.what() << "\n";
            return 1;
        }
        cfg.match_aware = true;
    }
    if (oracle_leaves && !search_model.find_method("oracle").has_value()) {
        std::cerr << "--oracle-leaves requires a search model traced with the oracle method\n";
        return 1;
    }
    if (!belief_path.empty()) {
        torch::jit::script::Module bm;
        try {
            bm = torch::jit::load(belief_path);
            bm.eval();
        } catch (const c10::Error& e) {
            std::cerr << "Failed to load belief model: " << e.what() << "\n";
            return 1;
        }
        if (ProbeObsDim(bm) != sdim) {
            std::cerr << "Belief model obs width must match the search model (" << sdim << ")\n";
            return 1;
        }
        cfg.belief_backend = std::make_shared<DirectBackend>(std::move(bm), device);
    }
    std::unique_ptr<IPlayer> sp;
    if (use_tree) {
        TreeSearchPlayer::Config tcfg;
        tcfg.iterations = tree_iterations;
        tcfg.c_puct = tree_c_puct;
        tcfg.seed = seed + 1000;
        tcfg.pass_cfg = cfg;  // sampler/pass search inherit device, K, belief settings
        sp = std::make_unique<TreeSearchPlayer>(search_model, sdim, tcfg);
    } else {
        sp = std::make_unique<SearchPlayer>(search_model, sdim, cfg);
    }
    RawPolicy opp(opp_model, odim);

    if (!match_path.empty() || !b_search_path.empty()) {
        // Paired matches to 100: table A = the (possibly match-aware) search
        // player at the test seat; table B = EITHER a second search player
        // (--b-search-model, e.g. the frozen match-blind reference - the
        // validation design) OR the 556-dim match-aware raw net
        // (--match-model, the original bridge measurement). Both arms play
        // identical deal-sequence seeds vs three anchor seats and, per rules
        // #15, share the same K schedule.
        std::unique_ptr<MatchRawPolicy> mp;
        std::unique_ptr<SearchPlayer> b_sp;
        if (!b_search_path.empty()) {
            torch::jit::script::Module bm;
            try {
                bm = torch::jit::load(b_search_path);
                bm.eval();
            } catch (const c10::Error& e) {
                std::cerr << "Failed to load B search model: " << e.what() << "\n";
                return 1;
            }
            int bdim = ProbeObsDim(bm);
            if (bdim == 0) {
                std::cerr << "B search model rejected all known obs widths\n";
                return 1;
            }
            SearchPlayer::Config bcfg = cfg;   // same K schedule, seeds offset
            bcfg.match_aware = false;          // arm B is the match-BLIND reference
            bcfg.equity_model.reset();
            bcfg.probe_log.clear();
            bcfg.seed = seed + 777000;
            b_sp = std::make_unique<SearchPlayer>(bm, bdim, bcfg);
        } else {
            torch::jit::script::Module mm;
            try {
                mm = torch::jit::load(match_path);
                mm.eval();
            } catch (const c10::Error& e) {
                std::cerr << "Failed to load match model: " << e.what() << "\n";
                return 1;
            }
            if (ProbeObsDim(mm) != 556) {
                std::cerr << "--match-model must be a 556-dim match trace\n";
                return 1;
            }
            mp = std::make_unique<MatchRawPolicy>(std::move(mm));
        }

        SearchPlayer* sp_flat = use_tree ? nullptr
                                         : static_cast<SearchPlayer*>(sp.get());
        if (!probe_log.empty() && sp_flat == nullptr) {
            std::cerr << "--probe-log requires the flat search player (no --tree)\n";
            return 2;
        }

        auto play_match = [&](HeartsEnv& env, std::array<double, 4>& totals,
                              int& deals_played, int test_seat, bool search_side) {
            while (true) {
                if (search_side && sp_flat) {
                    sp_flat->SetMatchContext(totals, deals_played);
                }
                if (!search_side && b_sp) {
                    b_sp->SetMatchContext(totals, deals_played);
                }
                env.Reset();
                bool done = false;
                while (!done) {
                    int p = env.GetCurrentPlayer();
                    int action;
                    if (p != test_seat) action = opp.ChooseAction(env);
                    else if (search_side) action = sp->ChooseAction(env);
                    else if (b_sp) action = b_sp->ChooseAction(env);
                    else action = mp->ChooseAction(env, totals, deals_played);
                    done = env.Step(action).done;
                }
                auto rs = env.GetRoundScores();
                for (int i2 = 0; i2 < 4; ++i2) totals[i2] += rs[i2];
                ++deals_played;
                if (deals_played >= 60 ||
                    *std::max_element(totals.begin(), totals.end()) >= 100.0) break;
            }
        };

        if (!behave_totals.empty()) {
            // Behavioral diagnostic: paired SINGLE deals from a constructed
            // score state (test seat's totals given as "self,left,across,
            // right" relative to the test seat). Emits per-deal round scores
            // for both arms; behavior metrics computed offline.
            std::array<double, 4> rel{};
            if (sscanf(behave_totals.c_str(), "%lf,%lf,%lf,%lf",
                       &rel[0], &rel[1], &rel[2], &rel[3]) != 4) {
                std::cerr << "--behave-totals wants self,left,across,right\n";
                return 2;
            }
            if (!b_sp) {
                std::cerr << "--behave requires --b-search-model\n";
                return 2;
            }
            std::ofstream bcsv(out_path);
            bcsv << "deal,seat,a0,a1,a2,a3,b0,b1,b2,b3\n";
            int deals_ctx = (int)std::round(
                (rel[0] + rel[1] + rel[2] + rel[3]) / 26.0);
            for (int di = 0; di < behave_deals; ++di) {
                int seat = di % 4;
                std::array<double, 4> totals{};
                for (int i = 0; i < 4; ++i) totals[(seat + i) % 4] = rel[i];
                unsigned dseed = seed + (unsigned)di * 1000u;
                std::array<std::array<int, 4>, 2> rs{};
                for (int arm = 0; arm < 2; ++arm) {
                    HeartsEnv env(dseed, true);
                    // align pass rotation with the implied deal count
                    for (int r = 0; r < deals_ctx % 4; ++r) env.Reset();
                    env.Reset();
                    if (arm == 0 && sp_flat) sp_flat->SetMatchContext(totals, deals_ctx);
                    if (arm == 1 && b_sp) b_sp->SetMatchContext(totals, deals_ctx);
                    bool done = false;
                    while (!done) {
                        int p = env.GetCurrentPlayer();
                        int action;
                        if (p != seat) action = opp.ChooseAction(env);
                        else if (arm == 0) action = sp->ChooseAction(env);
                        else action = b_sp->ChooseAction(env);
                        done = env.Step(action).done;
                    }
                    auto s = env.GetRoundScores();
                    for (int i = 0; i < 4; ++i) rs[arm][i] = s[i];
                }
                bcsv << di << "," << seat;
                for (int arm = 0; arm < 2; ++arm)
                    for (int i = 0; i < 4; ++i) bcsv << "," << rs[arm][i];
                bcsv << "\n";
                if ((di + 1) % 20 == 0)
                    std::cerr << "behave deal " << (di + 1) << "/" << behave_deals << "\n";
            }
            bcsv.close();
            std::cout << "behave complete, results " << out_path << "\n";
            return 0;
        }

        std::ofstream mcsv(out_path);
        mcsv << "match,seat,deals_a,final_a,place_a,win_a,deals_b,final_b,place_b,win_b\n";
        double sum_dp = 0.0;
        int wins_a = 0, wins_b = 0;
        auto mt0 = std::chrono::steady_clock::now();

        for (int mi = 0; mi < matches; ++mi) {
            int seat = mi % 4;
            unsigned mseed = seed + (unsigned)mi * 1000u;
            std::array<double, 4> ta{}, tb{};
            int da = 0, db = 0;
            HeartsEnv ea(mseed, true), eb(mseed, true);
            play_match(ea, ta, da, seat, true);
            play_match(eb, tb, db, seat, false);
            auto pa = Placements(ta), pb = Placements(tb);
            sum_dp += pa[seat] - pb[seat];
            wins_a += (pa[seat] == 1.0);
            wins_b += (pb[seat] == 1.0);
            mcsv << mi << "," << seat << "," << da << "," << ta[seat] << ","
                 << pa[seat] << "," << (pa[seat] == 1.0 ? 1 : 0) << ","
                 << db << "," << tb[seat] << "," << pb[seat] << ","
                 << (pb[seat] == 1.0 ? 1 : 0) << "\n";
            if ((mi + 1) % 5 == 0) {
                auto el = std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::steady_clock::now() - mt0).count();
                std::cerr << "match " << (mi + 1) << "/" << matches
                          << "  mean place diff (search-raw) " << (sum_dp / (mi + 1))
                          << "  wins " << wins_a << ":" << wins_b
                          << "  elapsed " << el << "s\n";
            }
        }
        mcsv.close();
        std::cout << "matches " << matches
                  << " mean_place_diff_search_minus_raw " << (sum_dp / matches)
                  << " wins_search " << wins_a << " wins_raw " << wins_b << "\n";
        std::cout << "results " << out_path << "\n";
        return 0;
    }

    HeartsEnv env_a(seed, true), env_b(seed, true);
    std::ofstream csv(out_path);
    csv << "deal,seat,score_a,score_b,diff\n";

    auto t0 = std::chrono::steady_clock::now();
    double sum_diff = 0.0;
    for (int d = 0; d < deals; ++d) {
        int seat = d % 4;

        env_a.Reset();
        bool done = false;
        while (!done) {
            int p = env_a.GetCurrentPlayer();
            int action = (p == seat) ? sp->ChooseAction(env_a) : opp.ChooseAction(env_a);
            done = env_a.Step(action).done;
        }
        auto sa = env_a.GetRoundScores();

        env_b.Reset();
        done = false;
        while (!done) {
            done = env_b.Step(opp.ChooseAction(env_b)).done;
        }
        auto sb = env_b.GetRoundScores();

        int diff = sa[seat] - sb[seat];
        sum_diff += diff;
        csv << d << "," << seat << "," << sa[seat] << "," << sb[seat] << "," << diff << "\n";

        if ((d + 1) % 10 == 0) {
            auto el = std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::steady_clock::now() - t0).count();
            std::cerr << "deal " << (d + 1) << "/" << deals
                      << "  running mean diff " << (sum_diff / (d + 1))
                      << "  elapsed " << el << "s\n";
        }
    }
    csv.close();
    std::cout << "mean_diff " << (sum_diff / deals) << "\n";
    std::cout << "results " << out_path << "\n";
    return 0;
}
