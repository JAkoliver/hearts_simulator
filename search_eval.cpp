// SearchEval: paired-deal evaluation of the decision-time search player,
// plus a --selftest mode validating the search machinery.
//
// Usage:
//   SearchEval --search-model <pt> --selftest
//   SearchEval --search-model <pt> --opponent-model <pt> --deals 300 --k 32
//              --seed 42 [--uniform-sampling] --out results.csv

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "HeartsEnv.hpp"
#include "SearchPlayer.hpp"

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

static bool TestDeterminizations(torch::jit::script::Module& model, int obs_dim) {
    SearchPlayer::Config cfg;
    cfg.determinizations = 8;
    cfg.seed = 99;
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

static bool TestBatchEquivalence(torch::jit::script::Module& model, int obs_dim) {
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
    torch::Tensor batched = model.forward({ob, mb}).toTuple()->elements()[0].toTensor();
    for (int i = 0; i < n; ++i) {
        torch::Tensor single = model.forward({ob.slice(0, i, i + 1), mb.slice(0, i, i + 1)})
                                   .toTuple()->elements()[0].toTensor();
        torch::Tensor mask_row = mb.slice(0, i, i + 1);
        double diff = (batched.slice(0, i, i + 1).masked_select(mask_row)
                       - single.masked_select(mask_row)).abs().max().item<double>();
        if (diff > 1e-4) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    std::string search_path, opp_path, out_path = "search_eval_results.csv";
    int deals = 300, k = 32;
    unsigned int seed = 42;
    bool uniform = false, selftest = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--search-model") search_path = next();
        else if (a == "--opponent-model") opp_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--deals") deals = std::stoi(next());
        else if (a == "--k") k = std::stoi(next());
        else if (a == "--seed") seed = static_cast<unsigned int>(std::stoul(next()));
        else if (a == "--uniform-sampling") uniform = true;
        else if (a == "--selftest") selftest = true;
        else { std::cerr << "Unknown arg: " << a << "\n"; return 2; }
    }
    if (search_path.empty()) {
        std::cerr << "--search-model is required\n";
        return 2;
    }

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
        t = TestDeterminizations(search_model, sdim);
        std::cerr << (t ? "PASS" : "FAIL") << "\n";
        ok &= t;
        std::cerr << "Batched-vs-single inference... ";
        t = TestBatchEquivalence(search_model, sdim);
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
    SearchPlayer sp(search_model, sdim, cfg);
    RawPolicy opp(opp_model, odim);

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
            int action = (p == seat) ? sp.ChooseAction(env_a) : opp.ChooseAction(env_a);
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
