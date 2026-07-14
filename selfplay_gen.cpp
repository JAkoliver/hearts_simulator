// SelfPlayGen: expert-iteration data generator.
//
// Plays search-vs-search self-play games (all four seats are the search
// teacher, including passing-phase search) and writes one training record per
// decision:
//
//   record (818 bytes, little-endian, no padding):
//     550 x u8   observation, quantized x*255 (all dims lie in [0,1])
//      52 x u8   legal-action mask
//     156 x u8   belief labels (true opponent hands, relative seats)
//      52 x u8   teacher policy target, quantized p*255 (softmax over the
//                search's per-action mean values; one-hot for pass picks and
//                forced moves) - soft targets keep distillation robust to
//                determinization noise on near-tie decisions
//       u16      action chosen by the search teacher
//       u16      acting seat
//       f32      relative round reward for the acting seat (avg - own score)
//
// Usage:
//   SelfPlayGen --model hearts_ai_search.pt --deals 1000 --k 16
//               --pass-k 12 --pass-candidates 12 --seed 1 --out selfplay_1.bin

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <torch/script.h>

#include "HeartsEnv.hpp"
#include "SearchPlayer.hpp"

struct PendingRec {
    std::array<uint8_t, 550> obs;
    std::array<uint8_t, 52> mask;
    std::array<uint8_t, 156> labels;
    std::array<uint8_t, 52> pi;
    uint16_t action;
    uint16_t seat;
};

int main(int argc, char** argv) {
    std::string model_path, out_path = "selfplay.bin";
    int deals = 1000, k = 16, pass_k = 12, pass_candidates = 12;
    unsigned int seed = 1;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--model") model_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--deals") deals = std::stoi(next());
        else if (a == "--k") k = std::stoi(next());
        else if (a == "--pass-k") pass_k = std::stoi(next());
        else if (a == "--pass-candidates") pass_candidates = std::stoi(next());
        else if (a == "--seed") seed = static_cast<unsigned int>(std::stoul(next()));
        else { std::cerr << "Unknown arg: " << a << "\n"; return 2; }
    }
    if (model_path.empty()) {
        std::cerr << "--model is required\n";
        return 2;
    }

    // Many SelfPlayGen processes run in parallel; without this each one
    // spawns LibTorch threads on every core and they thrash each other.
    torch::set_num_threads(1);

    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Failed to load model: " << e.what() << "\n";
        return 1;
    }
    int dim = ProbeObsDim(model);
    if (dim != 550) {
        std::cerr << "SelfPlayGen expects a current-generation (550-dim) search model, got "
                  << dim << "\n";
        return 1;
    }

    std::vector<SearchPlayer> players;
    for (int p = 0; p < 4; ++p) {
        SearchPlayer::Config cfg;
        cfg.determinizations = k;
        cfg.belief_weighted = true;
        cfg.pass_search = true;
        cfg.pass_k = pass_k;
        cfg.pass_candidates = pass_candidates;
        cfg.seed = seed * 7919u + p * 104729u;
        players.emplace_back(model, dim, cfg);
    }

    HeartsEnv env(seed, true);
    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::cerr << "Cannot open output file " << out_path << "\n";
        return 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    long total_records = 0;

    for (int d = 0; d < deals; ++d) {
        env.Reset();
        std::vector<PendingRec> recs;
        recs.reserve(64);
        bool done = false;

        while (!done) {
            int p = env.GetCurrentPlayer();
            auto obs = env.Observe();
            auto labels = env.ObserveOpponentHands();
            auto legal_raw = env.GetLegalActions();

            int action = players[p].ChooseAction(env);

            PendingRec r;
            for (int i = 0; i < 550; ++i) {
                float v = obs[i];
                if (v < 0.0f) v = 0.0f;
                if (v > 1.0f) v = 1.0f;
                r.obs[i] = static_cast<uint8_t>(std::lround(v * 255.0f));
            }
            r.mask.fill(0);
            for (int i = 0; i < 13; ++i) {
                if (legal_raw[i] != -1) r.mask[legal_raw[i]] = 1;
            }
            for (int i = 0; i < 156; ++i) {
                r.labels[i] = labels[i] > 0.5f ? 1 : 0;
            }
            const auto& pi = players[p].LastPolicy();
            for (int i = 0; i < 52; ++i) {
                r.pi[i] = static_cast<uint8_t>(std::lround(pi[i] * 255.0f));
            }
            r.action = static_cast<uint16_t>(action);
            r.seat = static_cast<uint16_t>(p);
            recs.push_back(r);

            done = env.Step(action).done;
        }

        auto sc = env.GetRoundScores();
        float avg = (sc[0] + sc[1] + sc[2] + sc[3]) / 4.0f;
        for (const auto& r : recs) {
            float reward = avg - sc[r.seat];
            out.write(reinterpret_cast<const char*>(r.obs.data()), 550);
            out.write(reinterpret_cast<const char*>(r.mask.data()), 52);
            out.write(reinterpret_cast<const char*>(r.labels.data()), 156);
            out.write(reinterpret_cast<const char*>(r.pi.data()), 52);
            out.write(reinterpret_cast<const char*>(&r.action), 2);
            out.write(reinterpret_cast<const char*>(&r.seat), 2);
            out.write(reinterpret_cast<const char*>(&reward), 4);
        }
        total_records += static_cast<long>(recs.size());

        if ((d + 1) % 5 == 0 || d + 1 == deals) {
            auto el = std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::steady_clock::now() - t0).count();
            std::cerr << "deal " << (d + 1) << "/" << deals
                      << "  records " << total_records
                      << "  elapsed " << el << "s\n";
        }
    }

    out.close();
    std::cout << "records " << total_records << "\n";
    std::cout << "file " << out_path << "\n";
    return 0;
}
