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
// --threads N runs N deal-playing threads in ONE process. With --cuda they
// share a single InferenceServer that coalesces every waiting request into
// one large forward - one CUDA context, few big launches instead of many
// small ones (separate processes serialize at the GPU and waste it).
// Each thread writes its own file: --out selfplay.bin -> selfplay_t0.bin ...
//
// Usage:
//   SelfPlayGen --model hearts_ai_search.pt --deals 6000 --k 64
//               --pass-k 24 --pass-candidates 12 --seed 1 --threads 12
//               --cuda --out selfplay.bin

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <torch/cuda.h>
#include <torch/script.h>

#include "HeartsEnv.hpp"
#include "InferenceServer.hpp"
#include "SearchPlayer.hpp"

struct PendingRec {
    std::array<uint8_t, 550> obs;
    std::array<uint8_t, 52> mask;
    std::array<uint8_t, 156> labels;
    std::array<uint8_t, 52> pi;
    uint16_t action;
    uint16_t seat;
};

struct GenShared {
    std::atomic<long> deals_done{0};
    std::atomic<long> records{0};
    std::atomic<bool> failed{false};
    long total_deals = 0;
    std::chrono::steady_clock::time_point t0;
};

static std::string ThreadOutPath(const std::string& base, int tid, int threads) {
    if (threads == 1) return base;
    size_t dot = base.find_last_of('.');
    if (dot == std::string::npos) return base + "_t" + std::to_string(tid);
    return base.substr(0, dot) + "_t" + std::to_string(tid) + base.substr(dot);
}

static void RunWorker(int tid, int deals, unsigned int seed,
                      std::shared_ptr<InferenceBackend> backend, int dim,
                      const SearchPlayer::Config& base_cfg,
                      const std::string& out_path, GenShared& shared) {
    try {
        std::vector<SearchPlayer> players;
        for (int p = 0; p < 4; ++p) {
            SearchPlayer::Config cfg = base_cfg;
            cfg.seed = seed * 7919u + p * 104729u;
            players.emplace_back(backend, dim, cfg);
        }

        HeartsEnv env(seed, true);
        std::ofstream out(out_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Cannot open output file " + out_path);
        }

        for (int d = 0; d < deals; ++d) {
            if (shared.failed.load()) return;  // another thread died; stop early
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
            shared.records.fetch_add(static_cast<long>(recs.size()));
            long total_done = shared.deals_done.fetch_add(1) + 1;

            if (total_done % 25 == 0 || total_done == shared.total_deals) {
                auto el = std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::steady_clock::now() - shared.t0).count();
                std::cerr << "deal " << total_done << "/" << shared.total_deals
                          << "  records " << shared.records.load()
                          << "  elapsed " << el << "s\n";
            }
        }
        out.close();
    } catch (const std::exception& e) {
        shared.failed.store(true);
        std::cerr << "worker thread " << tid << " failed: " << e.what() << "\n";
    }
}

int main(int argc, char** argv) {
    std::string model_path, out_path = "selfplay.bin";
    int deals = 1000, k = 16, pass_k = 12, pass_candidates = 12, threads = 1;
    unsigned int seed = 1;
    bool use_cuda = false;

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
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--cuda") use_cuda = true;
        else { std::cerr << "Unknown arg: " << a << "\n"; return 2; }
    }
    if (model_path.empty()) {
        std::cerr << "--model is required\n";
        return 2;
    }
    if (use_cuda && !torch::cuda::is_available()) {
        std::cerr << "--cuda requested but CUDA is not available in this libtorch build\n";
        return 1;
    }
    if (threads < 1) threads = 1;

    // The engine simulation runs on the worker threads; LibTorch's own pools
    // must not fan out on top of them.
    torch::set_num_threads(1);

    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Failed to load model: " << e.what() << "\n";
        return 1;
    }
    int dim = ProbeObsDim(model);  // probe on CPU, before any device move
    if (dim != 550) {
        std::cerr << "SelfPlayGen expects a current-generation (550-dim) search model, got "
                  << dim << "\n";
        return 1;
    }

    torch::Device device = use_cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    // One inference funnel for all threads: a coalescing server on the GPU,
    // or a single shared direct backend on CPU (forward is thread-safe and
    // runs on the calling thread).
    std::unique_ptr<InferenceServer> server;
    std::shared_ptr<InferenceBackend> backend;
    if (use_cuda) {
        server = std::make_unique<InferenceServer>(model, device);
        backend = std::make_shared<ServedBackend>(server.get());
    } else {
        backend = std::make_shared<DirectBackend>(model, device);
    }

    SearchPlayer::Config base_cfg;
    base_cfg.determinizations = k;
    base_cfg.belief_weighted = true;
    base_cfg.pass_search = true;
    base_cfg.pass_k = pass_k;
    base_cfg.pass_candidates = pass_candidates;

    GenShared shared;
    shared.total_deals = deals;
    shared.t0 = std::chrono::steady_clock::now();

    int per_thread = deals / threads;
    int extra = deals % threads;
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
        int quota = per_thread + (t < extra ? 1 : 0);
        if (quota == 0) continue;
        pool.emplace_back(RunWorker, t, quota, seed + t, backend, dim, base_cfg,
                          ThreadOutPath(out_path, t, threads), std::ref(shared));
    }
    for (auto& th : pool) th.join();

    if (shared.failed.load()) return 1;

    if (server) {
        std::cerr << "inference launches " << server->Launches()
                  << "  mean batch rows " << server->MeanBatchRows() << "\n";
    }
    std::cout << "records " << shared.records.load() << "\n";
    std::cout << "file " << out_path << (threads > 1 ? " (per-thread suffixes)" : "") << "\n";
    return 0;
}
