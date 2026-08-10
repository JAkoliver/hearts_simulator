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

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "HeartsEnv.hpp"
#include "SearchPlayer.hpp"
#include "TreeSearchPlayer.hpp"

// Lossless pause/resume for --shooter generation (round-2 prereg
// amendment): every CSV row and v2 .sdrec record carries its match index,
// and matches are generated sequentially - so after a kill, at most the
// LAST match is partial. Trim everything from the first incomplete match
// onward and return that index as the restart point. A match is complete
// when someone reached 100 or it hit the deal cap.
static int ShooterResumeTrim(const std::string& csv_path,
                             const std::string& tricks_path,
                             const std::string& rec_path, int deal_cap) {
    const char* DHDR = "match,seed,seat,deal,pass_dir,play_dec,alive_dec,"
                       "shoot_dec,pass_committed,pass_moonp,pass_eq_shoot,"
                       "pass_eq_norm,first_dead_trick,moon_success,"
                       "defender_moon,s0,s1,s2,s3,t0,t1,t2,t3\n";
    const char* THDR = "match,deal,trick,winner,points,shooter_seat\n";
    std::map<int, int> deal_rows;
    std::map<int, double> max_total;
    std::vector<std::string> rows;
    {
        std::ifstream in(csv_path);
        std::string line;
        bool first = true;
        while (in && std::getline(in, line)) {
            if (first) { first = false; continue; }   // header
            if (line.empty()) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            std::string tok;
            while (std::getline(ss, tok, ',')) f.push_back(tok);
            if (f.size() < 23) continue;
            int m = std::atoi(f[0].c_str());
            double mx = 0;
            for (size_t j = f.size() - 4; j < f.size(); ++j)
                mx = std::max(mx, std::atof(f[j].c_str()));
            deal_rows[m]++;
            max_total[m] = std::max(max_total[m], mx);
            rows.emplace_back(line);
        }
    }
    int mi0 = 0;
    while (true) {
        auto it = deal_rows.find(mi0);
        if (it == deal_rows.end()) break;
        if (!(max_total[mi0] >= 100.0 - 1e-9 || it->second >= deal_cap)) break;
        ++mi0;
    }
    {   // rewrite deal CSV: header + rows of complete matches only
        std::ofstream out(csv_path);
        out << DHDR;
        for (const auto& r : rows)
            if (std::atoi(r.c_str()) < mi0) out << r << '\n';
    }
    {   // tricks CSV: same trim (creates with header if absent)
        std::vector<std::string> trows;
        {
            std::ifstream in(tricks_path);
            std::string line;
            bool first = true;
            while (in && std::getline(in, line)) {
                if (first) { first = false; continue; }
                if (!line.empty() && std::atoi(line.c_str()) < mi0)
                    trows.emplace_back(line);
            }
        }
        std::ofstream out(tricks_path);
        out << THDR;
        for (const auto& r : trows) out << r << '\n';
    }
    if (!rec_path.empty()) {   // v2 records: keep prefix with match < mi0
        std::ifstream in(rec_path, std::ios::binary);
        std::string buf;
        if (in) {
            std::stringstream ss;
            ss << in.rdbuf();
            buf = ss.str();
        }
        in.close();
        const size_t REC = 2284, MOFF = 2224 + 52 + 4 + 1 + 1;
        size_t keep = 0;
        while (keep + REC <= buf.size()) {
            uint16_t m;
            std::memcpy(&m, buf.data() + keep + MOFF, 2);
            if ((int)m >= mi0) break;
            keep += REC;
        }
        std::ofstream out(rec_path, std::ios::binary);
        out.write(buf.data(), (std::streamsize)keep);
    }
    return mi0;
}

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


// Lossless resume for --tree-selfplay (Phase 2 prereg): records are a
// fixed 2522 bytes; a match ends with a kind=1 trailer. Resume trims
// the file to the last complete match and returns how many are done.
static int TreeResumeTrim(const std::string& record_path) {
    std::ifstream f(record_path, std::ios::binary | std::ios::ate);
    if (!f) return 0;
    const long long REC = 2522;
    long long n = f.tellg() / REC;
    int completed = 0;
    long long keep = 0;
    for (long long i = 0; i < n; ++i) {
        f.seekg(i * REC + 2493);   // kind byte
        char kind = 0;
        f.read(&kind, 1);
        if (kind == 1) {
            completed++;
            keep = (i + 1) * REC;
        }
    }
    f.close();
    std::filesystem::resize_file(record_path, keep);
    return completed;
}

int main(int argc, char** argv) {
    // Engine work is plain C++ on this thread; without this, libtorch's
    // intra-op pool fans tiny CPU ops across every core - harmless on a
    // 16-thread desktop, but on a 224-thread cloud box the spin-wait
    // coordination cost STALLED the process entirely (measured 2026-07-17:
    // 0 deals in 25 min vs 0.95 s/deal with the pool pinned).
    torch::set_num_threads(1);
    std::string search_path, opp_path, belief_path, match_path, probe_log,
        b_search_path, equity_path, behave_totals, shooter_flag, tricks_path,
        record_path, out_path = "search_eval_results.csv";
    int behave_deals = 200;
    int deals = 300, k = 32, rollout_tricks = -1, matches = 200, probe_every = 5,
        k_endgame = 0;
    int tree_iterations = 400;
    float tree_c_puct = 1.5f;
    unsigned int seed = 42;
    bool uniform = false, selftest = false, pass_search = false, use_cuda = false;
    bool oracle_leaves = false, use_tree = false, use_bf16 = false;
    bool tree_selfplay = false;
    std::string flat_compare;
    int compare_k = 64;
    int compare_seed_off = 5000;
    bool search_defenders = false;
    bool resume_flag = false;
    std::string attacker_path;

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
        else if (a == "--shooter") shooter_flag = next();
        else if (a == "--search-defenders") search_defenders = true;
        else if (a == "--resume") resume_flag = true;
        else if (a == "--attacker-model") attacker_path = next();
        else if (a == "--tricks-out") tricks_path = next();
        else if (a == "--record-out") record_path = next();
        else if (a == "--deals") deals = std::stoi(next());
        else if (a == "--k") k = std::stoi(next());
        else if (a == "--seed") seed = static_cast<unsigned int>(std::stoul(next()));
        else if (a == "--rollout-tricks") rollout_tricks = std::stoi(next());
        else if (a == "--oracle-leaves") oracle_leaves = true;
        else if (a == "--tree") use_tree = true;
        else if (a == "--tree-selfplay") tree_selfplay = true;
        else if (a == "--flat-compare") flat_compare = next();
        else if (a == "--compare-k") compare_k = std::stoi(next());
        else if (a == "--compare-seed-off") compare_seed_off = std::stoi(next());
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


    if (tree_selfplay) {
        // Phase 2 generation (docs/phase2_visitcount_prereg.md): the
        // match-aware TREE teacher plays ALL FOUR seats of natural
        // matches; every play decision is recorded with the root visit
        // distribution (the distillation target). Passing delegates to
        // the flat pass search and is NOT recorded (prereg scope).
        // Record layout (little-endian, 2522 bytes fixed):
        //   f32 obs[556] | u8 mask[52] | f32 pi[52] | i32 action |
        //   i32 total_visits | u8 seat | u8 kind (0 play, 1 trailer) |
        //   u16 match | u8 deal | u8 pad | i16 final[4] | f32 place[4]
        // Per-deal flush; --resume trims to the last trailer (the r2
        // kill-anytime contract).
        if (sdim != 556) {
            std::cerr << "--tree-selfplay requires a 556-dim search model\n";
            return 2;
        }
        if (equity_path.empty() || record_path.empty()) {
            std::cerr << "--tree-selfplay requires --equity-model and --record-out\n";
            return 2;
        }
        std::shared_ptr<torch::jit::script::Module> eq;
        try {
            eq = std::make_shared<torch::jit::script::Module>(
                torch::jit::load(equity_path));
            eq->eval();
        } catch (const c10::Error& e) {
            std::cerr << "Failed to load equity model: " << e.what() << "\n";
            return 1;
        }
        SearchPlayer::Config pc;
        pc.determinizations = k;
        pc.seed = seed + 1000;
        pc.pass_search = true;
        pc.device = device;
        pc.bf16 = use_bf16;
        pc.equity_model = eq;
        pc.match_aware = true;
        TreeSearchPlayer::Config tc;
        tc.iterations = tree_iterations;
        tc.c_puct = tree_c_puct;
        tc.seed = seed;
        tc.root_noise = true;    // self-play diversity (deterministic per seed)
        tc.equity_model = eq;
        tc.pass_cfg = pc;
        TreeSearchPlayer tsp(std::move(search_model), 556, tc);

        // Stage B dual probe (--flat-compare): the FLAT search evaluates
        // the SAME live position after the tree; per-decision CSV pairs
        // the tree's visit-share gap with the flat value gap - the
        // prereg's preference-strength validity check.
        std::unique_ptr<SearchPlayer> cmp;
        std::ofstream cmp_out;
        if (!flat_compare.empty()) {
            torch::jit::script::Module m2;
            try {
                m2 = torch::jit::load(search_path);
                m2.eval();
            } catch (const c10::Error& e) {
                std::cerr << "flat-compare model reload failed: "
                          << e.what() << "\n";
                return 1;
            }
            SearchPlayer::Config pc2 = pc;
            pc2.determinizations = compare_k;
            // Stage B-2: the K=256 rerun must draw an INDEPENDENT
            // determinization stream from the K=64 run (same base seed,
            // shared prefix would correlate the two references' noise
            // and inflate the reliability estimate).
            pc2.seed = seed + compare_seed_off;
            pc2.pass_search = false;
            cmp = std::make_unique<SearchPlayer>(std::move(m2), 556, pc2);
            cmp_out.open(flat_compare);
            cmp_out << "match,deal,seat,legal_n,tree_top1,tree_top2,"
                       "pi1,pi2,flat_best_v,flat_second_v,"
                       "flat_v_tree1,flat_v_tree2\n";
        }

        int mi0 = 0;
        if (resume_flag) {
            mi0 = TreeResumeTrim(record_path);
            std::cerr << "resume: trimmed to " << mi0
                      << " complete matches; continuing from match "
                      << mi0 << "\n";
        }
        const bool fresh = (mi0 == 0 && !resume_flag);
        std::ofstream rec(record_path, fresh
            ? std::ios::binary
            : (std::ios::binary | std::ios::out | std::ios::app));

        auto put_rec = [&](const float* obs556, const uint8_t* mask52,
                           const float* pi52, int32_t action,
                           int32_t visits, uint8_t seat, uint8_t kind,
                           uint16_t match, uint8_t deal,
                           const int16_t* fin4, const float* place4) {
            rec.write((const char*)obs556, 556 * 4);
            rec.write((const char*)mask52, 52);
            rec.write((const char*)pi52, 52 * 4);
            rec.write((const char*)&action, 4);
            rec.write((const char*)&visits, 4);
            rec.write((const char*)&seat, 1);
            rec.write((const char*)&kind, 1);
            rec.write((const char*)&match, 2);
            rec.write((const char*)&deal, 1);
            uint8_t pad = 0;
            rec.write((const char*)&pad, 1);
            rec.write((const char*)fin4, 8);
            rec.write((const char*)place4, 16);
        };

        long total_plays = 0;
        int total_deals = 0;
        auto pt0 = std::chrono::steady_clock::now();
        for (int mi = mi0; mi < matches; ++mi) {
            unsigned mseed = seed + (unsigned)mi * 1000u;
            HeartsEnv env(mseed, true);
            std::array<double, 4> totals{};
            int deals = 0;
            while (true) {
                tsp.SetMatchContext(totals, deals);
                env.Reset();
                bool done = false;
                while (!done) {
                    int p = env.GetCurrentPlayer();
                    bool was_passing = env.IsPassing();
                    int action = tsp.ChooseAction(env);
                    if (!was_passing) {
                        float row[556] = {};
                        auto obs = env.Observe();
                        std::memcpy(row, obs.data(), 550 * sizeof(float));
                        double mx = *std::max_element(totals.begin(),
                                                      totals.end());
                        for (int i2 = 0; i2 < 4; ++i2)
                            row[550 + i2] =
                                (float)(totals[(p + i2) % 4] / 100.0);
                        row[554] = (float)(deals / 20.0);
                        row[555] = (float)((100.0 - mx) / 100.0);
                        uint8_t mk[52] = {};
                        auto lr = env.GetLegalActions();
                        for (int i2 = 0; i2 < 13; ++i2)
                            if (lr[i2] != -1) mk[lr[i2]] = 1;
                        const auto& pi = tsp.LastPolicy();
                        int legal_n = 0;
                        for (int i2 = 0; i2 < 52; ++i2) legal_n += mk[i2];
                        // forced moves never ran the tree: visits would
                        // be STALE from the previous search - record 0
                        int32_t vis = legal_n > 1 ? tsp.LastRootVisits() : 0;
                        int16_t z4[4] = {};
                        float zf[4] = {};
                        put_rec(row, mk, pi.data(), action,
                                vis, (uint8_t)p, 0,
                                (uint16_t)mi, (uint8_t)deals, z4, zf);
                        total_plays++;
                        if (cmp && legal_n > 1) {
                            cmp->SetMatchContext(totals, deals);
                            cmp->ChooseAction(env);
                            const auto& st2 = cmp->LastStats();
                            if (st2.valid && st2.actions.size() > 1) {
                                int t1 = -1, t2 = -1;
                                float p1 = -1, p2v = -1;
                                for (int a2 = 0; a2 < 52; ++a2) {
                                    if (pi[a2] > p1) {
                                        p2v = p1; t2 = t1;
                                        p1 = pi[a2]; t1 = a2;
                                    } else if (pi[a2] > p2v) {
                                        p2v = pi[a2]; t2 = a2;
                                    }
                                }
                                float b1 = -1e9f, b2 = -1e9f;
                                float vt1 = 0, vt2 = 0;
                                for (size_t ai = 0; ai < st2.actions.size(); ++ai) {
                                    float v = st2.mean[ai];
                                    if (v > b1) { b2 = b1; b1 = v; }
                                    else if (v > b2) b2 = v;
                                    if (st2.actions[ai] == t1) vt1 = v;
                                    if (st2.actions[ai] == t2) vt2 = v;
                                }
                                cmp_out << mi << ',' << deals << ',' << p
                                        << ',' << legal_n << ',' << t1
                                        << ',' << t2 << ',' << p1 << ','
                                        << p2v << ',' << b1 << ',' << b2
                                        << ',' << vt1 << ',' << vt2 << '\n';
                                cmp_out.flush();
                            }
                        }
                    }
                    done = env.Step(action).done;
                }
                auto sc = env.GetRoundScores();
                for (int i2 = 0; i2 < 4; ++i2) totals[i2] += sc[i2];
                total_deals++;
                deals++;
                rec.flush();
                if (deals >= 60 ||
                    *std::max_element(totals.begin(), totals.end()) >= 100.0)
                    break;
            }
            float place[4];
            int16_t fin[4];
            for (int s2 = 0; s2 < 4; ++s2) {
                double better = 0, tied = 0;
                for (int o = 0; o < 4; ++o) {
                    if (o == s2) continue;
                    if (totals[o] < totals[s2]) better += 1;
                    else if (totals[o] == totals[s2]) tied += 1;
                }
                place[s2] = (float)(1.0 + better + tied * 0.5);
                fin[s2] = (int16_t)totals[s2];
            }
            float zrow[556] = {};
            uint8_t zmk[52] = {};
            float zpi[52] = {};
            put_rec(zrow, zmk, zpi, -1, 0, 0, 1, (uint16_t)mi,
                    (uint8_t)deals, fin, place);
            rec.flush();
            auto el = std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::steady_clock::now() - pt0).count();
            std::cerr << "tree match " << (mi + 1) << "/" << matches
                      << "  deals " << total_deals << "  plays "
                      << total_plays << "  elapsed " << el << "s\n";
        }
        std::cout << "tree-selfplay matches " << (matches - mi0)
                  << " deals " << total_deals << " plays " << total_plays
                  << "\nrecords " << record_path << "\n";
        return 0;
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

    if (!shooter_flag.empty()) {
        // Moon-shooter instrument (exploiter league Phase A,
        // docs/exploiter_league_prereg.md): the deployed match-aware search
        // stack with moon scoring - equity model, pass search, flat player.
        if (equity_path.empty() || !pass_search || use_tree) {
            std::cerr << "--shooter requires --equity-model and --pass-search"
                         " (and no --tree)\n";
            return 2;
        }
    }
    SearchPlayer::Config cfg;
    cfg.determinizations = k;
    cfg.belief_weighted = !uniform;
    cfg.seed = seed + 1000;
    cfg.pass_search = pass_search;
    cfg.shooter_mode = shooter_flag;
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
        // match-aware tree (556 trace) needs the equity leaves here too -
        // without this the deal-scoring driver can't host the Stage C
        // teacher config (P2 strength screen, 2026-08-10)
        tcfg.equity_model = cfg.equity_model;
        sp = std::make_unique<TreeSearchPlayer>(search_model, sdim, tcfg);
    } else {
        sp = std::make_unique<SearchPlayer>(search_model, sdim, cfg);
    }
    RawPolicy opp(opp_model, odim);

    if (!shooter_flag.empty()) {
        // ------------------- Phase A: shooter vs defender field -------------------
        // One search-shooter seat (rotating mi % 4) vs three copies of the
        // defender net (--opponent-model: 556 match trace plays with match
        // context, 550/238 raw plays match-blind), matches to 100.
        //
        // ROBUSTNESS: raw per-trick events (winner + points from the trick's
        // CARD CONTENTS - GetRoundScores is corrupted for trick 13, where
        // Step applies the 0/26/26/26 moon adjustment in the same call) plus
        // per-deal rows, flushed per deal. Metric definitions (threat-alive,
        // defense cost, interventions) are computed OFFLINE from the events
        // (analyze_phasea.py), so they can be refined without re-running
        // search-speed matches.
        SearchPlayer* sp_flat = static_cast<SearchPlayer*>(sp.get());
        std::unique_ptr<MatchRawPolicy> mdef;
        if (odim == 556) mdef = std::make_unique<MatchRawPolicy>(opp_model);
        // Round-2 teacher measurement (--search-defenders): the SAME search
        // program, standard match-equity scoring (shooter mode OFF), seated
        // in all three defender chairs. One instance serves all three: the
        // search context is rebuilt per decision from the acting seat, and
        // the 556 match ctx rotates by acting seat inside the row builder.
        std::unique_ptr<SearchPlayer> def_sp;
        if (search_defenders) {
            SearchPlayer::Config defcfg = cfg;
            defcfg.shooter_mode.clear();
            def_sp = std::make_unique<SearchPlayer>(search_model, sdim, defcfg);
        }
        // Round-2 corpus mode (--attacker-model): a distilled shooter CLONE
        // (556 match trace, argmax - its certified behavior) plays the
        // shooter seat instead of the search-shooter. Threat-alive then has
        // no shooter internals to read - the harness tracks it directly:
        // the moon is alive while no NON-attacker seat has taken a point.
        std::unique_ptr<MatchRawPolicy> matt;
        if (!attacker_path.empty()) {
            try {
                auto am = torch::jit::load(attacker_path);
                am.eval();
                matt = std::make_unique<MatchRawPolicy>(std::move(am));
            } catch (const c10::Error& e) {
                std::cerr << "Failed to load attacker model: " << e.what()
                          << "\n";
                return 1;
            }
        }
        // DECISION recorder v2 (round 2, docs/exploiter_league_r2_prereg.md):
        // one fixed-size record per recorded decision - shooter-seat
        // decisions always; with --search-defenders the three defender
        // chairs too (play AND pass picks). Layout, little-endian:
        //   float32 obs[556]  (engine 550 + match ctx, acting-seat view)
        //   uint8   mask[52]  (legal actions)
        //   int32   action    (the search player's choice)
        //   uint8   flags     (bit0 pass, bit1 threat-alive, bit2 shooting,
        //                      bit3 defender; alive/shooting are harness
        //                      labels for sample SELECTION - the obs stays
        //                      info-honest)
        //   uint8   seat      (acting seat)
        //   uint16  match     (match index - the lossless-resume key)
        // = 2284 bytes/record (same size as v1; pad became seat+match).
        // Flushed per deal.
        const std::string tpath = tricks_path.empty() ? out_path + ".tricks.csv"
                                                      : tricks_path;
        const char* DHDR = "match,seed,seat,deal,pass_dir,play_dec,alive_dec,"
                           "shoot_dec,pass_committed,pass_moonp,pass_eq_shoot,"
                           "pass_eq_norm,first_dead_trick,moon_success,"
                           "defender_moon,s0,s1,s2,s3,t0,t1,t2,t3\n";
        const char* THDR = "match,deal,trick,winner,points,shooter_seat\n";
        int mi0 = 0;
        if (resume_flag) {
            mi0 = ShooterResumeTrim(out_path, tpath, record_path, 60);
            std::cerr << "resume: trimmed to " << mi0
                      << " complete matches; continuing from match " << mi0
                      << "\n";
        }
        const bool fresh = (mi0 == 0 && !resume_flag);
        std::ofstream dcsv(out_path, fresh ? std::ios::out
                                           : (std::ios::out | std::ios::app));
        std::ofstream tcsv(tpath, fresh ? std::ios::out
                                        : (std::ios::out | std::ios::app));
        if (fresh) {
            dcsv << DHDR;
            tcsv << THDR;
        }
        std::ofstream rec;
        if (!record_path.empty()) {
            rec.open(record_path, fresh
                ? std::ios::binary
                : (std::ios::binary | std::ios::out | std::ios::app));
        }
        int moons = 0, committed_deals = 0, total_deals = 0;
        auto pt0 = std::chrono::steady_clock::now();

        for (int mi = mi0; mi < matches; ++mi) {
            int seat = mi % 4;
            unsigned mseed = seed + (unsigned)mi * 1000u;
            HeartsEnv env(mseed, true);
            std::array<double, 4> totals{};
            int deals = 0;
            while (true) {
                sp_flat->SetMatchContext(totals, deals);
                env.Reset();
                int pdir = env.GetPassDirection();
                int play_dec = 0, alive_dec = 0, shoot_dec = 0;
                int pass_committed = -1;   // -1 = hold deal (no pass decision)
                float pass_moonp = 0.0f, pass_eqs = 0.0f, pass_eqn = 0.0f;
                int first_dead = -1;
                int other_pts = 0;   // points taken by non-attacker seats
                bool pass_seen = false;
                bool done = false;
                while (!done) {
                    int p = env.GetCurrentPlayer();
                    bool was_passing = env.IsPassing();
                    int action;
                    // v2 record writer: obs/mask read from the CURRENT state
                    // (ChooseAction never mutates the env), ctx rotated by
                    // the acting seat, match id = the resume key.
                    auto write_rec = [&](int actor, int act, uint8_t flags) {
                        float row[556] = {};
                        uint8_t mk[52] = {};
                        auto obs = env.Observe();
                        std::memcpy(row, obs.data(), 550 * sizeof(float));
                        double mx = *std::max_element(totals.begin(),
                                                      totals.end());
                        for (int i2 = 0; i2 < 4; ++i2) {
                            row[550 + i2] =
                                (float)(totals[(actor + i2) % 4] / 100.0);
                        }
                        row[554] = (float)(deals / 20.0);
                        row[555] = (float)((100.0 - mx) / 100.0);
                        auto lr = env.GetLegalActions();
                        for (int i2 = 0; i2 < 13; ++i2) {
                            if (lr[i2] != -1) mk[lr[i2]] = 1;
                        }
                        int32_t act32 = act;
                        uint8_t seat8 = (uint8_t)actor;
                        uint16_t m16 = (uint16_t)mi;
                        rec.write((const char*)row, sizeof(row));
                        rec.write((const char*)mk, sizeof(mk));
                        rec.write((const char*)&act32, 4);
                        rec.write((const char*)&flags, 1);
                        rec.write((const char*)&seat8, 1);
                        rec.write((const char*)&m16, 2);
                    };
                    if (p == seat) {
                        if (matt) {   // clone attacker: no search, no stats
                            action = matt->ChooseAction(env, totals, deals);
                        } else {
                        action = sp->ChooseAction(env);
                        const auto& ss = sp_flat->LastShooter();
                        if (rec.is_open()) {
                            uint8_t flags = (was_passing ? 1 : 0)
                                | (ss.valid && ss.alive ? 2 : 0)
                                | (ss.valid && ss.shooting ? 4 : 0);
                            write_rec(p, action, flags);
                        }
                        if (ss.valid) {
                            if (was_passing) {
                                if (!pass_seen) {   // commit decided on pick 1
                                    pass_seen = true;
                                    pass_committed = ss.shooting ? 1 : 0;
                                    pass_moonp = ss.moon_p;
                                    pass_eqs = ss.eq_shoot;
                                    pass_eqn = ss.eq_norm;
                                }
                            } else {
                                play_dec++;
                                if (ss.alive) alive_dec++;
                                if (ss.shooting) shoot_dec++;
                            }
                        }
                        }   // end search-shooter path
                    } else if (def_sp) {
                        def_sp->SetMatchContext(totals, deals);
                        action = def_sp->ChooseAction(env);
                        // Defender curriculum (round 2): record every
                        // defender decision. alive is a harness LABEL for
                        // sample selection (the obs itself stays info-honest
                        // - defenders never see the shooter's internals).
                        // Clone attackers expose no internals: alive then
                        // means "no non-attacker seat has taken a point".
                        if (rec.is_open()) {
                            const auto& ss2 = sp_flat->LastShooter();
                            bool alive_lab = matt ? (other_pts == 0)
                                : (ss2.valid && ss2.alive);
                            uint8_t flags = (was_passing ? 1 : 0)
                                | (alive_lab ? 2 : 0)
                                | 8;
                            write_rec(p, action, flags);
                        }
                    } else if (mdef) {
                        action = mdef->ChooseAction(env, totals, deals);
                    } else {
                        action = opp.ChooseAction(env);
                    }
                    int tp = env.GetState().tricks_played;
                    done = env.Step(action).done;
                    if (env.GetState().tricks_played > tp) {
                        const auto& st = env.GetState();
                        int pts = 0;
                        for (const auto& pc : st.last_trick) {
                            if (pc.card.suit == Suit::Hearts) pts += 1;
                            else if (pc.card.suit == Suit::Spades && pc.card.rank == 12)
                                pts += 13;
                        }
                        int w = st.last_trick_winner;
                        tcsv << mi << ',' << deals << ',' << (tp + 1) << ','
                             << w << ',' << pts << ',' << seat << '\n';
                        if (w != seat) other_pts += pts;
                        if (first_dead < 0 && pts > 0 && w != seat)
                            first_dead = tp + 1;
                    }
                }
                auto sc = env.GetRoundScores();
                int stot = sc[0] + sc[1] + sc[2] + sc[3];
                bool success = (stot == 78 && sc[seat] == 0);
                bool dmoon = (stot == 78 && sc[seat] != 0);
                for (int i2 = 0; i2 < 4; ++i2) totals[i2] += sc[i2];
                dcsv << mi << ',' << mseed << ',' << seat << ',' << deals << ','
                     << pdir << ',' << play_dec << ',' << alive_dec << ','
                     << shoot_dec << ',' << pass_committed << ',' << pass_moonp
                     << ',' << pass_eqs << ',' << pass_eqn << ',' << first_dead
                     << ',' << (success ? 1 : 0) << ',' << (dmoon ? 1 : 0);
                for (int i2 = 0; i2 < 4; ++i2) dcsv << ',' << sc[i2];
                for (int i2 = 0; i2 < 4; ++i2) dcsv << ',' << totals[i2];
                dcsv << '\n';
                dcsv.flush();
                tcsv.flush();
                if (rec.is_open()) rec.flush();
                moons += success;
                committed_deals += (pass_committed == 1 || shoot_dec > 0);
                total_deals++;
                deals++;
                if (deals >= 60 ||
                    *std::max_element(totals.begin(), totals.end()) >= 100.0) break;
            }
            auto el = std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::steady_clock::now() - pt0).count();
            std::cerr << "shooter match " << (mi + 1) << "/" << matches
                      << "  deals " << total_deals << "  committed "
                      << committed_deals << "  moons " << moons
                      << "  elapsed " << el << "s\n";
        }
        dcsv.close();
        tcsv.close();
        std::cout << "shooter " << shooter_flag << " matches " << matches
                  << " deals " << total_deals << " committed " << committed_deals
                  << " moons " << moons << "\n";
        std::cout << "results " << out_path << "\n";
        return 0;
    }

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
            SearchPlayer::Config bcfg = cfg;   // same K schedule
            bcfg.match_aware = false;          // arm B is the match-BLIND reference
            bcfg.equity_model.reset();
            bcfg.probe_log.clear();
            // CRN (validation pre-registration): identical search seed so
            // determinization draws align wherever the arms' trajectories do
            bcfg.seed = cfg.seed;
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

        // Per-match stratum telemetry (pre-registered strata, rules #16):
        // computed at each deal START from the running totals. S2 = tension
        // reached (max>=85 AND top-two within 10), S1 = max>=85 only,
        // S3 = never reached max>=85 (degenerate; descriptive only).
        struct MatchStrata { int tension_deals = 0; bool tension = false;
                             bool max85 = false; };
        auto play_match = [&](HeartsEnv& env, std::array<double, 4>& totals,
                              int& deals_played, int test_seat, bool search_side,
                              MatchStrata* strata = nullptr) {
            while (true) {
                if (strata) {
                    std::array<double, 4> srt = totals;
                    std::sort(srt.begin(), srt.end(), std::greater<double>());
                    if (srt[0] >= 85.0) {
                        strata->max85 = true;
                        if (srt[0] - srt[1] <= 10.0) {
                            strata->tension = true;
                            ++strata->tension_deals;
                        }
                    }
                }
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
        mcsv << "match,seat,deals_a,final_a,place_a,win_a,deals_b,final_b,place_b,win_b,"
                "tens_a,max85_a,tdeals_a,tens_b,max85_b,tdeals_b\n";
        double sum_dp = 0.0;
        int wins_a = 0, wins_b = 0;
        auto mt0 = std::chrono::steady_clock::now();

        for (int mi = 0; mi < matches; ++mi) {
            int seat = mi % 4;
            unsigned mseed = seed + (unsigned)mi * 1000u;
            std::array<double, 4> ta{}, tb{};
            int da = 0, db = 0;
            HeartsEnv ea(mseed, true), eb(mseed, true);
            MatchStrata sa, sb;
            play_match(ea, ta, da, seat, true, &sa);
            play_match(eb, tb, db, seat, false, &sb);
            auto pa = Placements(ta), pb = Placements(tb);
            sum_dp += pa[seat] - pb[seat];
            wins_a += (pa[seat] == 1.0);
            wins_b += (pb[seat] == 1.0);
            mcsv << mi << "," << seat << "," << da << "," << ta[seat] << ","
                 << pa[seat] << "," << (pa[seat] == 1.0 ? 1 : 0) << ","
                 << db << "," << tb[seat] << "," << pb[seat] << ","
                 << (pb[seat] == 1.0 ? 1 : 0) << ","
                 << (sa.tension ? 1 : 0) << "," << (sa.max85 ? 1 : 0) << ","
                 << sa.tension_deals << ","
                 << (sb.tension ? 1 : 0) << "," << (sb.max85 ? 1 : 0) << ","
                 << sb.tension_deals << "\n";
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
