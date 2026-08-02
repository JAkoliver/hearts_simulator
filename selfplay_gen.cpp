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
// --value-out <file> additionally writes LEAF VALUE records (710 bytes):
//     550 x u8   observation from seat s's perspective at a completed-trick
//                boundary (the exact state distribution value-bootstrapped
//                search queries at truncated rollout leaves - including
//                seats that are NOT on turn, which decision records never
//                cover)
//     156 x u8   the true hands of s's three opponents, relative seats
//                (oracle-head input: at a search leaf the determinized hands
//                play this role)
//       f32      seat s's true final relative round reward (avg - own)
// One record per seat per boundary (tricks 1..12; trick 13 is terminal).
//
// --match switches to MATCH-MODE expert iteration: the quota unit (--deals)
// becomes MATCHES to 100 (60-deal cap), all four seats are match-aware
// SearchPlayers (556-dim trace + --equity-model leaf scoring, optional
// --k-endgame), and output is RECORD FORMAT v2
// (docs/expert_iter_v2_prereg.md). Every match-mode output file starts with a
// 32-byte header:
//     "HMR2" (4B) | version u16 (=2) | record_size u16 (=848) |
//     base_seed u32 (the --seed value) | thread_id u16 | zero-fill to 32B
// followed by 848-byte records (little-endian, no padding): the 824-byte v1
// fields in the same order -
//     obs u8[556] (550 engine dims + the acting seat's 6 match-context dims),
//     mask u8[52], labels u8[156], pi u8[52], action u16, seat u16,
//     reward f32 = (2.5 - tie-aware final placement) * 4 in {+6,+2,-2,-6}
// - then the per-decision search statistics (from SearchPlayer::LastStats):
//     eq_best f32     mean rollout equity of the chosen action
//     eq_second f32   best NON-chosen legal action's mean equity
//     gap_se f32      sqrt(se_best^2 + se_second^2), SE of the gap
//     second_action u16  action id of the second-best action (0xFFFF when no
//                        per-action stats exist: pass picks / forced moves)
//     n_dets u16      determinizations used (k vs --k-endgame; 0 = no search)
//     match_id u32    global match counter, unique across threads
//     flags u16       bit0 = passing-phase decision, bit1 = tension state at
//                     decision time (max total >= 85 AND top two within 10)
//     reserved u16    zero
// ALL decisions are recorded (confident or not) - confidence filtering
// happens at train time (distill.py --min-confidence), never at generation.
// Old 824-byte v1 files (no header) remain readable via distill.py's
// MATCH_RECORD dtype; new files are always v2.
// --start-totals self,left,across,right constructs every match's starting
// score state (rotated so the "self" slot cycles absolute seats per match);
// --start-deals overrides the implied round(sum/26) deal count.
//
// Usage:
//   SelfPlayGen --model hearts_ai_search.pt --deals 6000 --k 64
//               --pass-k 24 --pass-candidates 12 --seed 1 --threads 12
//               --cuda --out selfplay.bin

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

// Headroom pacing for the C++ generator. headroom.py's pace() only covers
// PYTHON workloads; SelfPlayGen previously ran flat-out and saturated the
// GPU, making the desktop unusable (2026-07-29/30). With HEARTS_HEADROOM=f
// (0<f<0.95) each worker sleeps f/(1-f) x its own decision time after every
// decision, yielding both GPU and CPU in short slices => ~f of wall time
// given back. Unset/0 = full throttle, unchanged behaviour.
static double HeadroomFraction() {
    if (const char* s = std::getenv("HEARTS_HEADROOM")) {
        const double v = std::atof(s);
        if (v > 0.0 && v < 0.95) return v;
    }
    return 0.0;
}

#include <torch/cuda.h>
#include <torch/script.h>

#include "HeartsEnv.hpp"
#include "InferenceServer.hpp"
#include "SearchPlayer.hpp"
#include "TreeSearchPlayer.hpp"

struct PendingRec {
    std::array<uint8_t, 550> obs;
    std::array<uint8_t, 52> mask;
    std::array<uint8_t, 156> labels;
    std::array<uint8_t, 52> pi;
    uint16_t action;
    uint16_t seat;
};

// --match mode record (format v2): 556-dim obs (550 engine dims + the acting
// seat's 6 match-context dims, exactly as SearchPlayer::WriteCtx feeds the
// net), match-placement reward, and per-decision search statistics. 848 bytes
// on disk (see the file-format comment above), written field-by-field so
// struct padding can never leak in.
struct MatchPendingRec {
    std::array<uint8_t, 556> obs;
    std::array<uint8_t, 52> mask;
    std::array<uint8_t, 156> labels;
    std::array<uint8_t, 52> pi;
    uint16_t action;
    uint16_t seat;
    // v2 per-decision search statistics
    float eq_best = 0.0f;
    float eq_second = 0.0f;
    float gap_se = 0.0f;
    uint16_t second_action = 0xFFFF;
    uint16_t n_dets = 0;
    uint16_t flags = 0;
};

// 1 = winner (lowest total); ties share the mean of their ranks.
// (Copied from search_eval.cpp - keep in sync.)
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
                      int tree_iterations, float tree_c_puct,
                      const std::string& out_path, const std::string& value_out_path,
                      GenShared& shared) {
    try {
        std::vector<std::unique_ptr<IPlayer>> players;
        for (int p = 0; p < 4; ++p) {
            SearchPlayer::Config cfg = base_cfg;
            cfg.seed = seed * 7919u + p * 104729u;
            if (tree_iterations > 0) {
                TreeSearchPlayer::Config tcfg;
                tcfg.iterations = tree_iterations;
                tcfg.c_puct = tree_c_puct;
                tcfg.seed = cfg.seed;
                tcfg.root_noise = true;  // self-play exploration diversity
                tcfg.pass_cfg = cfg;
                players.push_back(std::make_unique<TreeSearchPlayer>(backend, dim, tcfg));
            } else {
                players.push_back(std::make_unique<SearchPlayer>(backend, dim, cfg));
            }
        }

        HeartsEnv env(seed, true);
        std::ofstream out(out_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Cannot open output file " + out_path);
        }
        std::ofstream vout;
        if (!value_out_path.empty()) {
            vout.open(value_out_path, std::ios::binary);
            if (!vout) {
                throw std::runtime_error("Cannot open value output file " + value_out_path);
            }
        }

        struct LeafRec {
            std::array<uint8_t, 550> obs;
            std::array<uint8_t, 156> hands;
            uint16_t seat;
        };

        for (int d = 0; d < deals; ++d) {
            if (shared.failed.load()) return;  // another thread died; stop early
            env.Reset();
            std::vector<PendingRec> recs;
            recs.reserve(64);
            std::vector<LeafRec> leaf_recs;
            int prev_tricks = 0;
            bool done = false;

            while (!done) {
                int p = env.GetCurrentPlayer();
                auto obs = env.Observe();
                auto labels = env.ObserveOpponentHands();
                auto legal_raw = env.GetLegalActions();

                int action = players[p]->ChooseAction(env);

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
                const auto& pi = players[p]->LastPolicy();
                for (int i = 0; i < 52; ++i) {
                    r.pi[i] = static_cast<uint8_t>(std::lround(pi[i] * 255.0f));
                }
                r.action = static_cast<uint16_t>(action);
                r.seat = static_cast<uint16_t>(p);
                recs.push_back(r);

                done = env.Step(action).done;

                // Leaf value records at every completed-trick boundary
                // (except the terminal one, whose value is the exact score)
                int tp = env.GetState().tricks_played;
                if (vout.is_open() && tp != prev_tricks && !done) {
                    for (int s = 0; s < 4; ++s) {
                        LeafRec lr;
                        auto lobs = env.ObserveFor(s);
                        for (int i = 0; i < 550; ++i) {
                            float v = lobs[i];
                            if (v < 0.0f) v = 0.0f;
                            if (v > 1.0f) v = 1.0f;
                            lr.obs[i] = static_cast<uint8_t>(std::lround(v * 255.0f));
                        }
                        auto lhands = env.ObserveOpponentHandsFor(s);
                        for (int i = 0; i < 156; ++i) {
                            lr.hands[i] = lhands[i] > 0.5f ? 1 : 0;
                        }
                        lr.seat = static_cast<uint16_t>(s);
                        leaf_recs.push_back(lr);
                    }
                }
                prev_tricks = tp;
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
            for (const auto& lr : leaf_recs) {
                float reward = avg - sc[lr.seat];
                vout.write(reinterpret_cast<const char*>(lr.obs.data()), 550);
                vout.write(reinterpret_cast<const char*>(lr.hands.data()), 156);
                vout.write(reinterpret_cast<const char*>(&reward), 4);
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

// --match mode: the quota unit is MATCHES to 100 (60-deal cap). Every seat is
// a match-aware SearchPlayer; the running totals/deal count are pushed into
// all four players before each deal (each seat's search then sees its OWN
// rotated context). One record per decision, buffered per match, reward
// assigned from the final tie-aware placements: (2.5 - place) * 4
// => {+6, +2, -2, -6} (ties: intermediate averages).
static void RunMatchWorker(int tid, int quota, long match_base, unsigned int seed,
                           unsigned int base_seed,
                           std::shared_ptr<InferenceBackend> backend, int dim,
                           const SearchPlayer::Config& base_cfg,
                           bool have_start, std::array<double, 4> start_rel,
                           int start_deals, int start_jitter,
                           const std::string& out_path,
                           GenShared& shared) {
    try {
        std::vector<std::unique_ptr<SearchPlayer>> players;
        for (int p = 0; p < 4; ++p) {
            SearchPlayer::Config cfg = base_cfg;
            cfg.seed = seed * 7919u + p * 104729u;
            players.push_back(std::make_unique<SearchPlayer>(backend, dim, cfg));
        }

        std::ofstream out(out_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Cannot open output file " + out_path);
        }
        // Record-format v2 file header (32 bytes, see the comment at the top).
        {
            char hdr[32] = {};
            std::memcpy(hdr, "HMR2", 4);
            const uint16_t version = 2, record_size = 848;
            const uint32_t bseed = base_seed;
            const uint16_t tid16 = static_cast<uint16_t>(tid);
            std::memcpy(hdr + 4, &version, 2);
            std::memcpy(hdr + 6, &record_size, 2);
            std::memcpy(hdr + 8, &bseed, 4);
            std::memcpy(hdr + 12, &tid16, 2);
            out.write(hdr, 32);
        }

        const double headroom = HeadroomFraction();
        std::vector<MatchPendingRec> recs;
        for (int m = 0; m < quota; ++m) {
            if (shared.failed.load()) return;  // another thread died; stop early

            // Constructed start state (behave-style): totals given relative to
            // a rotating seat so all four absolute seats see the "self" slot
            // equally often across matches.
            int rot_seat = static_cast<int>((match_base + m) % 4);
            std::array<double, 4> totals{};
            int deals_played = 0;
            if (have_start) {
                // --start-jitter J: per-match seed-derived offsets in [-J, +J]
                // per slot turn each seeded family from a single score vector
                // into a NEIGHBORHOOD on the score-state manifold (2026-08-01;
                // deterministic from (seed, m) - reproducibility preserved).
                // Clamp keeps totals in [0, 99] (still pre-elimination).
                std::array<double, 4> rel = start_rel;
                if (start_jitter > 0) {
                    std::mt19937 jrng(seed * 2654435761u +
                                      static_cast<unsigned>(m) * 40503u + 7u);
                    std::uniform_int_distribution<int> jd(-start_jitter,
                                                          start_jitter);
                    for (int i = 0; i < 4; ++i) {
                        double v = rel[i] + jd(jrng);
                        if (v < 0.0) v = 0.0;
                        if (v > 99.0) v = 99.0;
                        rel[i] = v;
                    }
                }
                for (int i = 0; i < 4; ++i) {
                    totals[(rot_seat + i) % 4] = rel[i];
                }
                deals_played = static_cast<int>(std::round(
                    (rel[0] + rel[1] + rel[2] + rel[3]) / 26.0));
            }
            if (start_deals >= 0) deals_played = start_deals;

            HeartsEnv env(seed + static_cast<unsigned>(m) * 1000u, true);
            // align pass rotation with the implied starting deal count
            for (int r = 0; r < deals_played % 4; ++r) env.Reset();

            recs.clear();
            while (true) {  // deals until the match ends
                for (int p = 0; p < 4; ++p) {
                    players[p]->SetMatchContext(totals, deals_played);
                }
                env.Reset();
                bool done = false;
                while (!done) {
                    int p = env.GetCurrentPlayer();
                    auto obs = env.Observe();
                    auto labels = env.ObserveOpponentHands();
                    auto legal_raw = env.GetLegalActions();

                    const auto d0 = std::chrono::steady_clock::now();
                    int action = players[p]->ChooseAction(env);
                    if (headroom > 0.0) {
                        const auto us = std::chrono::duration_cast<
                            std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - d0).count();
                        const auto nap = (long long)(us * headroom /
                                                     (1.0 - headroom));
                        if (nap > 0) {
                            std::this_thread::sleep_for(
                                std::chrono::microseconds(nap));
                        }
                    }

                    MatchPendingRec r;
                    for (int i = 0; i < 550; ++i) {
                        float v = obs[i];
                        if (v < 0.0f) v = 0.0f;
                        if (v > 1.0f) v = 1.0f;
                        r.obs[i] = static_cast<uint8_t>(std::lround(v * 255.0f));
                    }
                    // The acting seat's 6 match-context dims, exactly as
                    // SearchPlayer::WriteCtx builds them (seat-relative
                    // rotation, /100 and /20 scaling).
                    {
                        double mx = *std::max_element(totals.begin(), totals.end());
                        float ctx[6];
                        for (int i = 0; i < 4; ++i) {
                            ctx[i] = static_cast<float>(totals[(p + i) % 4] / 100.0);
                        }
                        ctx[4] = static_cast<float>(deals_played / 20.0);
                        ctx[5] = static_cast<float>((100.0 - mx) / 100.0);
                        for (int i = 0; i < 6; ++i) {
                            float v = ctx[i];
                            if (v < 0.0f) v = 0.0f;
                            if (v > 1.0f) v = 1.0f;
                            r.obs[550 + i] = static_cast<uint8_t>(std::lround(v * 255.0f));
                        }
                    }
                    r.mask.fill(0);
                    for (int i = 0; i < 13; ++i) {
                        if (legal_raw[i] != -1) r.mask[legal_raw[i]] = 1;
                    }
                    for (int i = 0; i < 156; ++i) {
                        r.labels[i] = labels[i] > 0.5f ? 1 : 0;
                    }
                    const auto& pi = players[p]->LastPolicy();
                    for (int i = 0; i < 52; ++i) {
                        r.pi[i] = static_cast<uint8_t>(std::lround(pi[i] * 255.0f));
                    }
                    r.action = static_cast<uint16_t>(action);
                    r.seat = static_cast<uint16_t>(p);

                    // v2 per-decision search statistics (defaults from the
                    // struct: zeros, second_action = 0xFFFF, flags = 0).
                    const SearchPlayer::ActionStats& st = players[p]->LastStats();
                    if (st.pass_phase) r.flags |= 1u;
                    {   // tension at decision time: max >= 85, top two within 10
                        std::array<double, 4> t = totals;
                        std::sort(t.begin(), t.end());  // ascending: t[3]=max
                        if (t[3] >= 85.0 && t[3] - t[2] <= 10.0) r.flags |= 2u;
                    }
                    if (st.valid) {
                        r.n_dets = static_cast<uint16_t>(st.n_dets);
                        int bi = -1, si = -1;
                        for (size_t i = 0; i < st.actions.size(); ++i) {
                            if (st.actions[i] == action) { bi = static_cast<int>(i); break; }
                        }
                        for (size_t i = 0; i < st.actions.size(); ++i) {
                            if (static_cast<int>(i) == bi) continue;
                            if (si < 0 || st.mean[i] > st.mean[si]) si = static_cast<int>(i);
                        }
                        if (bi >= 0 && si >= 0) {
                            r.eq_best = st.mean[bi];
                            r.eq_second = st.mean[si];
                            r.gap_se = std::sqrt(st.se[bi] * st.se[bi]
                                                 + st.se[si] * st.se[si]);
                            r.second_action = static_cast<uint16_t>(st.actions[si]);
                        }
                    }
                    recs.push_back(r);

                    done = env.Step(action).done;
                }
                auto rs = env.GetRoundScores();
                for (int i = 0; i < 4; ++i) totals[i] += rs[i];
                ++deals_played;
                if (deals_played >= 60 ||
                    *std::max_element(totals.begin(), totals.end()) >= 100.0) break;
            }

            auto place = Placements(totals);
            const uint32_t match_id = static_cast<uint32_t>(match_base + m);
            const uint16_t reserved = 0;
            for (const auto& r : recs) {
                float reward = static_cast<float>((2.5 - place[r.seat]) * 4.0);
                // 824 v1 fields in the v1 order, then the v2 tail: 848 bytes.
                out.write(reinterpret_cast<const char*>(r.obs.data()), 556);
                out.write(reinterpret_cast<const char*>(r.mask.data()), 52);
                out.write(reinterpret_cast<const char*>(r.labels.data()), 156);
                out.write(reinterpret_cast<const char*>(r.pi.data()), 52);
                out.write(reinterpret_cast<const char*>(&r.action), 2);
                out.write(reinterpret_cast<const char*>(&r.seat), 2);
                out.write(reinterpret_cast<const char*>(&reward), 4);
                out.write(reinterpret_cast<const char*>(&r.eq_best), 4);
                out.write(reinterpret_cast<const char*>(&r.eq_second), 4);
                out.write(reinterpret_cast<const char*>(&r.gap_se), 4);
                out.write(reinterpret_cast<const char*>(&r.second_action), 2);
                out.write(reinterpret_cast<const char*>(&r.n_dets), 2);
                out.write(reinterpret_cast<const char*>(&match_id), 4);
                out.write(reinterpret_cast<const char*>(&r.flags), 2);
                out.write(reinterpret_cast<const char*>(&reserved), 2);
            }
            shared.records.fetch_add(static_cast<long>(recs.size()));
            long total_done = shared.deals_done.fetch_add(1) + 1;

            if (total_done % 2 == 0 || total_done == shared.total_deals) {
                auto el = std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::steady_clock::now() - shared.t0).count();
                std::cerr << "match " << total_done << "/" << shared.total_deals
                          << "  records " << shared.records.load()
                          << "  elapsed " << el << "s\n";
            }
        }
        out.close();
    } catch (const std::exception& e) {
        shared.failed.store(true);
        std::cerr << "match worker thread " << tid << " failed: " << e.what() << "\n";
    }
}

int main(int argc, char** argv) {
    std::string model_path, out_path = "selfplay.bin", value_out_path;
    int deals = 1000, k = 16, pass_k = 12, pass_candidates = 12, threads = 1;
    int rollout_tricks = -1;
    int tree_iterations = 0;  // 0 = flat search teacher; >0 = ISMCTS tree teacher
    float tree_c_puct = 1.5f;
    bool oracle_leaves = false;
    unsigned int seed = 1;
    bool use_cuda = false, use_bf16 = false;
    // --match mode (match-aware expert iteration)
    bool match_mode = false;
    std::string equity_path, start_totals_str;
    int k_endgame = 0, start_deals = -1, start_jitter = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--model") model_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--value-out") value_out_path = next();
        else if (a == "--deals") deals = std::stoi(next());
        else if (a == "--k") k = std::stoi(next());
        else if (a == "--pass-k") pass_k = std::stoi(next());
        else if (a == "--pass-candidates") pass_candidates = std::stoi(next());
        else if (a == "--seed") seed = static_cast<unsigned int>(std::stoul(next()));
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--rollout-tricks") rollout_tricks = std::stoi(next());
        else if (a == "--oracle-leaves") oracle_leaves = true;
        else if (a == "--tree-iterations") tree_iterations = std::stoi(next());
        else if (a == "--tree-cpuct") tree_c_puct = std::stof(next());
        else if (a == "--cuda") use_cuda = true;
        else if (a == "--bf16") use_bf16 = true;
        else if (a == "--match") match_mode = true;
        else if (a == "--equity-model") equity_path = next();
        else if (a == "--k-endgame") k_endgame = std::stoi(next());
        else if (a == "--start-totals") start_totals_str = next();
        else if (a == "--start-deals") start_deals = std::stoi(next());
        else if (a == "--start-jitter") start_jitter = std::stoi(next());
        else { std::cerr << "Unknown arg: " << a << "\n"; return 2; }
    }
    if (model_path.empty()) {
        std::cerr << "--model is required\n";
        return 2;
    }
    bool have_start = false;
    std::array<double, 4> start_rel{};
    if (match_mode) {
        if (!value_out_path.empty()) {
            std::cerr << "--value-out is not supported with --match\n";
            return 2;
        }
        if (tree_iterations > 0) {
            std::cerr << "--match requires the flat search teacher (no --tree-iterations)\n";
            return 2;
        }
        if (equity_path.empty()) {
            std::cerr << "--match requires --equity-model\n";
            return 2;
        }
        if (rollout_tricks >= 0) {
            std::cerr << "--match requires full rollouts (drop --rollout-tricks)\n";
            return 2;
        }
        if (!start_totals_str.empty()) {
            if (std::sscanf(start_totals_str.c_str(), "%lf,%lf,%lf,%lf",
                            &start_rel[0], &start_rel[1], &start_rel[2],
                            &start_rel[3]) != 4) {
                std::cerr << "--start-totals wants self,left,across,right\n";
                return 2;
            }
            have_start = true;
        }
    } else if (!equity_path.empty() || k_endgame > 0 || !start_totals_str.empty()
               || start_deals >= 0) {
        std::cerr << "--equity-model/--k-endgame/--start-totals/--start-deals "
                     "require --match\n";
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
    if (match_mode) {
        if (dim != 556) {
            std::cerr << "--match expects a 556-dim match-aware search trace, got "
                      << dim << "\n";
            return 1;
        }
    } else if (dim != 550) {
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
        server = std::make_unique<InferenceServer>(model, device, use_bf16);
        backend = std::make_shared<ServedBackend>(server.get());
    } else {
        backend = std::make_shared<DirectBackend>(model, device, use_bf16);
    }

    SearchPlayer::Config base_cfg;
    base_cfg.determinizations = k;
    base_cfg.belief_weighted = true;
    base_cfg.pass_search = true;
    base_cfg.pass_k = pass_k;
    base_cfg.pass_candidates = pass_candidates;
    base_cfg.rollout_tricks = rollout_tricks;
    base_cfg.oracle_leaves = oracle_leaves;
    if (oracle_leaves && !model.find_method("oracle").has_value()) {
        std::cerr << "--oracle-leaves requires a search model traced with the oracle method\n";
        return 1;
    }
    if (match_mode) {
        try {
            base_cfg.equity_model = std::make_shared<torch::jit::script::Module>(
                torch::jit::load(equity_path));
            base_cfg.equity_model->eval();
        } catch (const c10::Error& e) {
            std::cerr << "Failed to load equity model: " << e.what() << "\n";
            return 1;
        }
        base_cfg.match_aware = true;
        base_cfg.k_endgame = k_endgame;
    }

    GenShared shared;
    shared.total_deals = deals;
    shared.t0 = std::chrono::steady_clock::now();

    int per_thread = deals / threads;
    int extra = deals % threads;
    std::vector<std::thread> pool;
    long match_base = 0;  // global match index of a thread's first match
    for (int t = 0; t < threads; ++t) {
        int quota = per_thread + (t < extra ? 1 : 0);
        if (quota == 0) continue;
        if (match_mode) {
            pool.emplace_back(RunMatchWorker, t, quota, match_base, seed + t,
                              seed, backend, dim, base_cfg, have_start,
                              start_rel, start_deals, start_jitter,
                              ThreadOutPath(out_path, t, threads),
                              std::ref(shared));
            match_base += quota;
        } else {
            std::string vpath = value_out_path.empty()
                                    ? std::string()
                                    : ThreadOutPath(value_out_path, t, threads);
            pool.emplace_back(RunWorker, t, quota, seed + t, backend, dim, base_cfg,
                              tree_iterations, tree_c_puct,
                              ThreadOutPath(out_path, t, threads), vpath, std::ref(shared));
        }
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
