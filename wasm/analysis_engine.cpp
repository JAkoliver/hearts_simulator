// Client-side match analysis engine (hearts_web/TODO.md: client-side search).
//
// Compiles the SAME HeartsEnv the server uses to WASM, plus a torch-free
// port of SearchPlayer's determinized full-rollout search (belief-weighted
// sampling, match-equity scoring). Inference is PUMPED, not embedded:
// the engine fills an obs/mask request buffer and returns control; the JS
// side runs onnxruntime-web asynchronously and feeds results back. No
// Asyncify, no libtorch.
//
//   an_load_match(seed, deal_action_offsets, actions)   full-replay contract
//   an_analyze(deal, action_idx, K)  -> pump kind
//   an_choose_pass(deal, seat, K, n_cand) -> pump kind (LIVE pass search)
//   an_pump kinds: 0 done, 1 policy rows wanted (root: feed logits+belief;
//                  rollout: feed argmax acts), 2 equity rows wanted
//
// Efficiency baked in (Phase-1 calibration): rollout rounds request only
// argmax actions (the 'act' graph output); FORCED moves (single legal
// card - endemic in Hearts endgames) are stepped engine-side with no net
// call at all; JS chunks any request to <=416 rows (the measured WebGPU
// cliff sits at 832).
//
// Faithfulness: BuildContext / TrySample / TrySampleExact / AssignSuits /
// ScoreEquity / TerminalWinValue are line-for-line ports from
// SearchPlayer.hpp (same constraints, same weighting, same units). Client
// results are near-parity, not bit-parity, with the native engine
// (different inference backend numerics) - the UI labels them with K.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../HeartsEnv.hpp"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define KEEP EMSCRIPTEN_KEEPALIVE
#else
#define KEEP
#endif

namespace {

constexpr int PUMP_DONE = 0;
constexpr int PUMP_POLICY = 1;   // rows in obs/mask; root wants logits+belief
constexpr int PUMP_EQUITY = 2;   // rows of 10 floats in eq_in; feed P(place1)

struct Sim {
    HeartsEnv env;
    int tag;                 // candidate-action index / combo / replica
    bool done = false;
    double result = 0.0;
    // Scripted pass picks (pass-combo analysis): while passing and it is
    // script_seat's turn, play from the script instead of the policy.
    std::vector<int> script;
    int script_seat = -1;
    size_t script_pos = 0;
    Sim(HeartsEnv e, int t) : env(std::move(e)), tag(t) {}
};

struct Engine {
    // ---- match replay ------------------------------------------------------
    // Hands are EXPLICIT per deal (52 ids, seat-major 4x13): seed-based
    // dealing is std::shuffle-implementation-bound, so a WASM (libc++)
    // build cannot reproduce an MSVC-recorded match from the seed alone -
    // measured 2026-08-05 on a real logged match.
    unsigned match_seed = 0;
    std::vector<std::vector<int>> deal_actions;   // per deal, raw action ids
    std::vector<std::array<std::vector<int>, 4>> deal_hands;

    // ---- decision state ----------------------------------------------------
    HeartsEnv env{1, true};
    bool env_valid = false;
    int me = 0;
    int deals_played = 0;
    std::array<double, 4> totals{};
    std::vector<int> legal;
    int K = 16;
    std::mt19937 rng{12345};

    // context (port of SearchPlayer::Context)
    std::vector<int> my_hand, unseen, pinned_rel;
    std::array<int, 3> cap{};
    std::array<std::array<bool, 4>, 3> is_void{};
    float belief[3][52] = {};

    // pump state
    int stage = PUMP_DONE;       // what the CURRENT request wants
    bool root_pending = false;
    std::vector<Sim> sims;
    std::vector<size_t> active;      // sims awaiting a policy act
    std::vector<size_t> eq_rows;     // sims awaiting an equity score
    std::vector<float> obs_buf;      // rows x 556
    std::vector<uint8_t> mask_buf;   // rows x 52
    std::vector<float> eq_buf;       // rows x 10
    std::vector<int32_t> act_buf;    // rows (JS writes)
    std::vector<float> f_in_buf;     // rows (JS writes equity P1)

    // results
    std::vector<int32_t> r_actions;
    std::vector<float> r_mean, r_se, r_pts;
    std::vector<int32_t> r_n;
    bool playout_mode = false;      // par playout: per-seat deal points
    bool trace_mode = false;        // par TRACE: one playout per play state
    bool pass_mode = false;         // pass-combo analysis
    bool cards_mode = false;        // cards-only match replicas
    std::array<int32_t, 4> r_po{};
    std::vector<int32_t> r_trace;   // n_states x 4 per-seat playout points
    std::vector<std::array<int, 3>> combos;   // pass candidates under eval
    std::array<int, 3> actual_combo{};        // the pass actually made
    bool have_actual = false;                 // review: anchor the actual pass; live: no actual yet
    int n_cand = 10;
    std::vector<int32_t> r_combo;   // n x 3 pass-combo card ids
    std::vector<int> cards_deal;    // per-replica current deal index
    std::vector<int32_t> r_cards;   // K x 6: totals[4], finished, deals

    // ---- helpers (ports) ---------------------------------------------------
    static std::vector<int> LegalVector(const HeartsEnv& e) {
        std::vector<int> out;
        auto lr = e.GetLegalActions();
        for (int i = 0; i < 13; ++i) {
            if (lr[i] != -1) out.push_back(lr[i]);
        }
        return out;
    }

    void WriteCtx(float* row, int seat) const {
        double mx = *std::max_element(totals.begin(), totals.end());
        for (int i = 0; i < 4; ++i) {
            row[550 + i] = (float)(totals[(seat + i) % 4] / 100.0);
        }
        row[554] = (float)(deals_played / 20.0);
        row[555] = (float)((100.0 - mx) / 100.0);
    }

    void FillObsRow(float* row, const HeartsEnv& e) const {
        auto obs = e.Observe();
        std::memcpy(row, obs.data(), 550 * sizeof(float));
        WriteCtx(row, e.GetCurrentPlayer());
    }

    void BuildContext() {
        my_hand.clear();
        unseen.clear();
        me = env.GetCurrentPlayer();
        const auto& played_by = env.GetPlayedBy();
        const auto& voids = env.GetVoidTracker();
        std::array<bool, 52> mine{};
        for (const auto& c : env.GetState().hands[me]) {
            int id = (int)c.suit * 13 + (c.rank - 2);
            mine[id] = true;
            my_hand.push_back(id);
        }
        for (int c = 0; c < 52; ++c) {
            bool played = played_by[0][c] || played_by[1][c] || played_by[2][c]
                          || played_by[3][c];
            if (!played && !mine[c]) unseen.push_back(c);
        }
        for (int k = 1; k < 4; ++k) {
            int abs_seat = (me + k) % 4;
            cap[k - 1] = env.GetHandSize(abs_seat);
            for (int s = 0; s < 4; ++s) {
                is_void[k - 1][s] = voids[abs_seat * 4 + s];
            }
        }
        pinned_rel.assign(52, -1);
        if (env.GetPassDirection() != 3) {
            int off = (env.GetPassDirection() == 0) ? 1
                      : (env.GetPassDirection() == 1) ? 3 : 2;
            for (int a : env.GetPassPicks(me)) {
                if (std::find(unseen.begin(), unseen.end(), a) != unseen.end())
                    pinned_rel[a] = off - 1;
            }
        }
    }

    bool TrySample(bool use_belief, std::array<std::vector<int>, 4>& out) {
        std::array<int, 3> caps = cap;
        std::vector<int> owner(52, -1);
        std::vector<int> todo;
        for (int c : unseen) {
            int pin = pinned_rel[c];
            if (pin >= 0) {
                if (caps[pin] <= 0 || is_void[pin][c / 13]) return false;
                owner[c] = pin;
                caps[pin]--;
            } else {
                todo.push_back(c);
            }
        }
        while (!todo.empty()) {
            int best_idx = -1, best_count = 4;
            for (size_t i = 0; i < todo.size(); ++i) {
                int c = todo[i], count = 0;
                for (int k = 0; k < 3; ++k) {
                    if (caps[k] > 0 && !is_void[k][c / 13]) count++;
                }
                if (count == 0) return false;
                if (count < best_count || (count == best_count && (rng() & 1))) {
                    best_count = count;
                    best_idx = (int)i;
                }
            }
            int c = todo[best_idx];
            todo.erase(todo.begin() + best_idx);
            double w[3], total = 0.0;
            for (int k = 0; k < 3; ++k) {
                bool ok = caps[k] > 0 && !is_void[k][c / 13];
                w[k] = ok ? (use_belief ? std::max(belief[k][c], 1e-4f) : 1.0)
                          : 0.0;
                total += w[k];
            }
            std::uniform_real_distribution<double> u(0.0, total);
            double r = u(rng);
            int pick = 2;
            for (int k = 0; k < 3; ++k) {
                if (r < w[k]) { pick = k; break; }
                r -= w[k];
            }
            if (w[pick] == 0.0) return false;
            owner[c] = pick;
            caps[pick]--;
        }
        out[me] = my_hand;
        for (int c = 0; c < 52; ++c) {
            if (owner[c] >= 0) out[(me + owner[c] + 1) % 4].push_back(c);
        }
        return true;
    }

    bool AssignSuits(int s, const std::array<int, 4>& need,
                     std::array<int, 3>& caps, int x[4][3]) {
        if (s == 4) return true;
        int n = need[s];
        int max0 = is_void[0][s] ? 0 : std::min(n, caps[0]);
        for (int i = 0; i <= max0; ++i) {
            int max1 = is_void[1][s] ? 0 : std::min(n - i, caps[1]);
            for (int j = 0; j <= max1; ++j) {
                int k = n - i - j;
                if (k > (is_void[2][s] ? 0 : caps[2])) continue;
                x[s][0] = i; x[s][1] = j; x[s][2] = k;
                caps[0] -= i; caps[1] -= j; caps[2] -= k;
                if (AssignSuits(s + 1, need, caps, x)) return true;
                caps[0] += i; caps[1] += j; caps[2] += k;
            }
        }
        return false;
    }

    bool TrySampleExact(std::array<std::vector<int>, 4>& out) {
        std::array<int, 3> caps = cap;
        std::vector<int> owner(52, -1);
        std::array<std::vector<int>, 4> suit_cards;
        for (int c : unseen) {
            int pin = pinned_rel[c];
            if (pin >= 0) {
                if (caps[pin] <= 0 || is_void[pin][c / 13]) return false;
                owner[c] = pin;
                caps[pin]--;
            } else {
                suit_cards[c / 13].push_back(c);
            }
        }
        std::array<int, 4> need{};
        for (int s = 0; s < 4; ++s) need[s] = (int)suit_cards[s].size();
        int x[4][3] = {};
        if (!AssignSuits(0, need, caps, x)) return false;
        for (int s = 0; s < 4; ++s) {
            std::shuffle(suit_cards[s].begin(), suit_cards[s].end(), rng);
            size_t i = 0;
            for (int k = 0; k < 3; ++k) {
                for (int j = 0; j < x[s][k]; ++j) owner[suit_cards[s][i++]] = k;
            }
        }
        out[me] = my_hand;
        for (int c = 0; c < 52; ++c) {
            if (owner[c] >= 0) out[(me + owner[c] + 1) % 4].push_back(c);
        }
        return true;
    }

    std::array<std::vector<int>, 4> SampleHands() {
        for (int attempt = 0; attempt < 200; ++attempt) {
            std::array<std::vector<int>, 4> hands;
            if (TrySample(attempt < 100, hands)) return hands;
        }
        std::array<std::vector<int>, 4> hands;
        if (TrySampleExact(hands)) return hands;
        // Unsatisfiable constraints = engine-state bug; surface as done/empty.
        return {};
    }

    // A default-constructed sample (all four hands empty) is SampleHands'
    // "unsatisfiable" sentinel. A valid determinization always holds at
    // least the acting seat's cards (play) or 13 everywhere (pass rewind).
    static bool DetEmpty(const std::array<std::vector<int>, 4>& h) {
        return h[0].empty() && h[1].empty() && h[2].empty() && h[3].empty();
    }

    // Fill `dets` with up to want valid determinizations, drawing at most
    // 3*want times. 2026-08-21 robustness fix (site follow-up): one
    // unlucky/unsatisfiable draw used to ride into SetHands /
    // ResetForPassSearch and abort the WHOLE search via Trap (-3) - the
    // caller lost all K determinizations instead of one. Skip bad draws;
    // only if fewer than max(4, want/4) survive is the position treated
    // as genuinely broken, with the surviving count in the message.
    void FillDets(std::vector<std::array<std::vector<int>, 4>>& out, int want) {
        int draws = 0;
        while ((int)out.size() < want && draws < 3 * want) {
            auto h = SampleHands();
            ++draws;
            if (!DetEmpty(h)) out.push_back(std::move(h));
        }
        int floor_n = want / 4 > 4 ? want / 4 : 4;
        if ((int)out.size() < floor_n) {
            throw std::runtime_error(
                "determinization starvation: " + std::to_string(out.size()) +
                " of " + std::to_string(want) + " samples valid");
        }
    }

    static double TerminalWinValue(const std::array<double, 4>& t, int seat) {
        double mine = t[seat];
        int better = 0, tied = 0;
        for (int i = 0; i < 4; ++i) {
            if (i == seat) continue;
            if (t[i] < mine) ++better;
            else if (t[i] == mine) ++tied;
        }
        if (better > 0) return 0.0;
        return 1.0 / (1 + tied);
    }

    // ---- replay ------------------------------------------------------------
    bool SeekTo(int deal_idx, int action_idx) {
        if (deal_idx >= (int)deal_actions.size()) return false;
        env = HeartsEnv(match_seed, true);
        totals = {0, 0, 0, 0};
        for (int d = 0; d < deal_idx; ++d) {
            env.Reset();
            if (d < (int)deal_hands.size()) env.SetDeal(deal_hands[d]);
            for (int a : deal_actions[d]) env.Step(a);
            auto sc = env.GetRoundScores();
            for (int i = 0; i < 4; ++i) totals[i] += sc[i];
        }
        deals_played = deal_idx;
        env.Reset();
        if (deal_idx < (int)deal_hands.size()) env.SetDeal(deal_hands[deal_idx]);
        const auto& acts = deal_actions[deal_idx];
        // action_idx == acts.size() is the PENDING decision (live play:
        // everything logged so far has been replayed, the decision itself
        // has not been made). Review callers always pass action_idx < size.
        if (action_idx > (int)acts.size()) return false;
        for (int i = 0; i < action_idx; ++i) env.Step(acts[i]);
        env_valid = true;
        return true;
    }

    // ---- pump --------------------------------------------------------------
    int RequestRoot() {
        obs_buf.assign(556, 0.0f);
        mask_buf.assign(52, 1);
        FillObsRow(obs_buf.data(), env);
        root_pending = true;
        stage = PUMP_POLICY;
        return stage;
    }

    void SpawnSims() {
        int start_deals = deals_played;
        (void)start_deals;
        sims.clear();
        // dets shared across actions (paired comparison); sampled up front
        // with skip-and-retry (FillDets) so one bad draw costs one det,
        // not the search. K_eff = surviving count.
        FillDets(dets, K);
        sims.reserve(legal.size() * dets.size());
        for (size_t ai = 0; ai < legal.size(); ++ai) {
            for (size_t d = 0; d < dets.size(); ++d) {
                Sim s(env.Clone(), (int)ai);
                s.env.SetHands(dets[d]);
                s.done = s.env.Step(legal[ai]).done;
                sims.push_back(std::move(s));
            }
        }
    }
    std::vector<std::array<std::vector<int>, 4>> dets;

    // Cards-only replicas play MULTIPLE deals: on deal end, roll into the
    // next dealt hand unless the match ended or hands ran out.
    void CardsAdvance(Sim& s) {
        if (!cards_mode || !s.done) return;
        const auto& tot = s.env.GetState().total_scores;
        int mx = *std::max_element(tot.begin(), tot.end());
        int next = cards_deal[s.tag] + 1;
        if (mx >= 100 || next >= (int)deal_hands.size()) return;  // stays done
        cards_deal[s.tag] = next;
        s.env.Reset();
        s.env.SetDeal(deal_hands[next]);
        s.done = false;
    }

    // Step every sim that needs no inference: forced single-legal moves.
    // Hearts rollouts are full of them (suit-following, late tricks) - each
    // one stepped here is a net call that never happens.
    void DrainForced() {
        bool progressed = true;
        while (progressed) {
            progressed = false;
            for (auto& s : sims) {
                if (s.done) continue;
                if (s.script_pos < s.script.size() && s.env.IsPassing()
                    && s.env.GetCurrentPlayer() == s.script_seat) {
                    s.done = s.env.Step(s.script[s.script_pos++]).done;
                    progressed = true;
                    continue;
                }
                auto lr = s.env.GetLegalActions();
                int only = -1, n = 0;
                for (int i = 0; i < 13 && n < 2; ++i) {
                    if (lr[i] != -1) { only = lr[i]; n++; }
                }
                if (n == 1) {
                    s.done = s.env.Step(only).done;
                    CardsAdvance(s);
                    progressed = true;
                }
            }
        }
    }

    int NextRolloutRequest() {
        DrainForced();
        active.clear();
        for (size_t i = 0; i < sims.size(); ++i) {
            if (!sims[i].done) active.push_back(i);
        }
        if (active.empty()) {
            if (cards_mode) {
                r_cards.assign(sims.size() * 6, 0);
                for (auto& s : sims) {
                    const auto& tot = s.env.GetState().total_scores;
                    int mx = *std::max_element(tot.begin(), tot.end());
                    for (int k2 = 0; k2 < 4; ++k2) {
                        r_cards[s.tag * 6 + k2] = tot[k2];
                    }
                    r_cards[s.tag * 6 + 4] = mx >= 100 ? 1 : 0;
                    r_cards[s.tag * 6 + 5] = cards_deal[s.tag] + 1;
                }
                sims.clear();
                stage = PUMP_DONE;
                return stage;
            }
            if (playout_mode || trace_mode) {
                r_trace.assign(sims.size() * 4, 0);
                for (auto& s : sims) {
                    auto sc = s.env.GetRoundScores();
                    for (int k = 0; k < 4; ++k) r_trace[s.tag * 4 + k] = sc[k];
                }
                if (playout_mode) {
                    for (int k = 0; k < 4; ++k) r_po[k] = r_trace[k];
                }
                sims.clear();
                stage = PUMP_DONE;
                return stage;
            }
            return RequestEquity();
        }
        obs_buf.assign(active.size() * 556, 0.0f);
        mask_buf.assign(active.size() * 52, 0);
        for (size_t j = 0; j < active.size(); ++j) {
            FillObsRow(obs_buf.data() + j * 556, sims[active[j]].env);
            auto lr = sims[active[j]].env.GetLegalActions();
            for (int i = 0; i < 13; ++i) {
                if (lr[i] != -1) mask_buf[j * 52 + lr[i]] = 1;
            }
        }
        act_buf.assign(active.size(), 0);
        stage = PUMP_POLICY;
        return stage;
    }

    int RequestEquity() {
        eq_rows.clear();
        const int deals_after = deals_played + 1;
        std::vector<float> rows;
        for (size_t i = 0; i < sims.size(); ++i) {
            auto sc = sims[i].env.GetRoundScores();
            std::array<double, 4> t = totals;
            for (int k = 0; k < 4; ++k) t[k] += sc[k];
            double mx = *std::max_element(t.begin(), t.end());
            if (mx >= 100.0) {
                sims[i].result = TerminalWinValue(t, me);
            } else {
                float row[10] = {};
                for (int k = 0; k < 4; ++k) {
                    row[k] = (float)(t[(me + k) % 4] / 100.0);
                }
                row[4] = (float)(deals_after / 20.0);
                row[5] = (float)((100.0 - mx) / 100.0);
                row[6 + (deals_after % 4)] = 1.0f;
                rows.insert(rows.end(), row, row + 10);
                eq_rows.push_back(i);
            }
        }
        if (eq_rows.empty()) return Finish();
        eq_buf = std::move(rows);
        f_in_buf.assign(eq_rows.size(), 0.0f);
        stage = PUMP_EQUITY;
        return stage;
    }

    int Finish() {
        size_t n = pass_mode ? combos.size() : legal.size();
        std::vector<double> sum(n, 0.0), sumsq(n, 0.0), psum(n, 0.0);
        std::vector<int> cnt(n, 0);
        for (const auto& s : sims) {
            sum[s.tag] += s.result;
            sumsq[s.tag] += s.result * s.result;
            // Deal points for the mover (post-moon-adjusted at deal end):
            // a concrete, human-readable second axis next to win chance.
            psum[s.tag] += s.env.GetRoundScores()[me];
            cnt[s.tag] += 1;
        }
        if (pass_mode) {
            r_actions.clear();
            r_combo.clear();
            for (const auto& c : combos) {
                r_actions.push_back(c[0]);
                for (int x : c) r_combo.push_back(x);
            }
        } else {
            r_actions.assign(legal.begin(), legal.end());
            r_combo.clear();
        }
        r_mean.assign(n, 0.0f);
        r_se.assign(n, 0.0f);
        r_pts.assign(n, 0.0f);
        r_n.assign(cnt.begin(), cnt.end());
        for (size_t i = 0; i < n; ++i) {
            int c = cnt[i];
            double mu = c > 0 ? sum[i] / c : 0.0;
            r_mean[i] = (float)mu;
            r_pts[i] = (float)(c > 0 ? psum[i] / c : 0.0);
            double se = 0.0;
            if (c >= 2) {
                double var = (sumsq[i] - c * mu * mu) / (c - 1);
                if (var < 0.0) var = 0.0;
                se = std::sqrt(var / c);
            }
            r_se[i] = (float)se;
        }
        sims.clear();
        dets.clear();
        stage = PUMP_DONE;
        return stage;
    }
};

Engine E;
char g_err[256] = {};

// A C++ throw in a default Emscripten build calls abort() and KILLS the
// worker thread with no message - the UI just freezes. Every entry point
// that can throw (env validation, sampling) is trapped instead: the pump
// returns -3 and an_error_msg() carries the reason to JS.
template <typename F>
int Trap(F&& f) {
    try {
        return f();
    } catch (const std::exception& e) {
        std::snprintf(g_err, sizeof(g_err), "%s", e.what());
        E.stage = PUMP_DONE;
        return -3;
    } catch (...) {
        std::snprintf(g_err, sizeof(g_err), "unknown C++ exception");
        E.stage = PUMP_DONE;
        return -3;
    }
}

}  // namespace

extern "C" {

KEEP const char* an_error_msg() { return g_err; }

KEEP void an_init(unsigned rng_seed) {
    E = Engine();
    E.rng.seed(rng_seed);
}

// Match replay: actions (deal boundaries via offsets, n_deals+1 entries)
// plus EXPLICIT start hands per deal (52 ids, seat-major 4x13; pass
// hands=null to fall back to seed dealing - same-toolchain only).
KEEP int an_load_match(unsigned seed, const int* offsets, int n_deals,
                       const int* actions, const int* hands) {
    E.match_seed = seed;
    E.deal_actions.assign(n_deals, {});
    E.deal_hands.clear();
    for (int d = 0; d < n_deals; ++d) {
        E.deal_actions[d].assign(actions + offsets[d], actions + offsets[d + 1]);
        if (hands) {
            std::array<std::vector<int>, 4> h;
            for (int s = 0; s < 4; ++s) {
                h[s].assign(hands + d * 52 + s * 13, hands + d * 52 + s * 13 + 13);
            }
            E.deal_hands.push_back(std::move(h));
        }
    }
    return n_deals;
}

// Begin analysis of one decision. Returns first pump kind (or -1 on bad
// coordinates / passing decision - v1 analyzes PLAY decisions only).
// PAR playout: from the deal's post-pass state, Perilune plays ALL FOUR
// seats to the deal's end (deterministic - true hands, argmax policy).
// The per-seat points are the deal's par: what these cards "should" cost.
KEEP int an_playout(int deal_idx) {
  return Trap([&]() -> int {
    E.sims.clear();
    E.dets.clear();
    E.playout_mode = true;
    E.trace_mode = E.pass_mode = E.cards_mode = false;
    E.rng.seed(0x7F4A7C15u ^ (unsigned)(deal_idx * 131071));
    int pass_actions =
        (deal_idx < (int)E.deal_actions.size()
         && E.deal_actions[deal_idx].size() == 64) ? 12 : 0;
    if (!E.SeekTo(deal_idx, pass_actions)) return -1;
    E.sims.emplace_back(E.env.Clone(), 0);
    E.root_pending = false;
    return E.NextRolloutRequest();
  });
}

KEEP int32_t* an_result_playout() { return E.r_po.data(); }

// PAR TRACE: one all-AI playout from the state BEFORE EVERY PLAY of one
// deal, batched into a single pump session (rows across sims share
// forwards). Client-side telescoping over the results attributes every
// point to the move that shifted it: own moves = unforced, others' =
// forced. V[j] rows are per-seat points of the playout from state j.
KEEP int an_deal_trace(int deal_idx) {
  return Trap([&]() -> int {
    E.sims.clear();
    E.dets.clear();
    E.playout_mode = E.pass_mode = E.cards_mode = false;
    E.trace_mode = true;
    E.rng.seed(0x2545F491u ^ (unsigned)(deal_idx * 131071));
    if (deal_idx >= (int)E.deal_actions.size()) return -1;
    const auto& acts = E.deal_actions[deal_idx];
    int pass_off = (int)acts.size() == 64 ? 12 : 0;
    if (!E.SeekTo(deal_idx, pass_off)) return -1;
    int n_plays = (int)acts.size() - pass_off;
    for (int j = 0; j < n_plays; ++j) {
        E.sims.emplace_back(E.env.Clone(), j);
        E.env.Step(acts[pass_off + j]);
    }
    E.root_pending = false;
    return E.NextRolloutRequest();
  });
}

KEEP int an_trace_n() { return (int)(E.r_trace.size() / 4); }
KEEP int32_t* an_result_trace() { return E.r_trace.data(); }

// PASS-COMBO analysis: evaluate candidate 3-card passes for `seat` on a
// passing deal, info-honestly from that seat's pre-pass view (own hand
// known, everything else determinized; the same rewound-pass machinery
// the native engine uses). Root wants logits+belief via
// an_feed_root_pass: logits propose candidate combos, belief weights the
// determinizations. The ACTUAL pass is always among the candidates.
// Shared pass-search setup. live=false (review): the deal is finished and
// the pass actually made is anchored as a candidate. live=true (play):
// only the picks of seats BEFORE this one exist in the log; no anchor -
// the candidate set is exactly the native SearchPlayer::ChoosePass one
// (TopThree + samples), scored combo x determinization.
static int BeginPass(int deal_idx, int seat, int k, int n_cand, bool live) {
    E.K = k;
    E.playout_mode = E.trace_mode = E.cards_mode = false;
    E.pass_mode = true;
    E.have_actual = false;
    E.n_cand = n_cand < 4 ? 4 : (n_cand > 20 ? 20 : n_cand);
    E.rng.seed((live ? 0x11FE5EEDu : 0x51ED270Fu)
               ^ (unsigned)(deal_idx * 131071 + seat * 257 + k));
    E.sims.clear();
    E.dets.clear();
    E.combos.clear();
    if (deal_idx >= (int)E.deal_actions.size()) return -1;
    const auto& acts = E.deal_actions[deal_idx];
    if (!live && (int)acts.size() != 64) return -1;        // review: hold deal has no pass
    if (live && (int)acts.size() < seat * 3) return -1;    // live: earlier seats' picks must be logged
    if (!E.SeekTo(deal_idx, seat * 3)) return -1;
    if (!E.env.IsPassing() || E.env.GetCurrentPlayer() != seat) return -1;
    E.me = seat;
    E.legal = Engine::LegalVector(E.env);
    if (!live) {
        for (int j = 0; j < 3; ++j) E.actual_combo[j] = acts[seat * 3 + j];
        std::sort(E.actual_combo.begin(), E.actual_combo.end());
        E.have_actual = true;
    }
    E.BuildContext();
    E.cap = {13, 13, 13};   // pass rewind: everyone back to full hands
    return E.RequestRoot();
}

KEEP int an_analyze_pass(int deal_idx, int seat, int k, int n_cand) {
  return Trap([&]() -> int { return BeginPass(deal_idx, seat, k, n_cand, false); });
}

// LIVE pass choice (site "search on" mode): same search as review minus the
// anchor; result via an_result_combo after the pump completes. Port of
// SearchPlayer::ChoosePass (native defaults pass_k=12 / pass_candidates=12;
// the teacher uses --pass-k 24).
KEEP int an_choose_pass(int deal_idx, int seat, int k, int n_cand) {
  return Trap([&]() -> int { return BeginPass(deal_idx, seat, k, n_cand, true); });
}

// Root feed for pass analysis: candidates from the policy's own pass
// distribution (top-3 + samples, ports of the native TopThree/SampleCombo)
// plus the actual pass; sims = combo x determinization, own picks
// scripted, everyone else by policy, scored like any other decision.
KEEP int an_feed_root_pass(const float* logits52, const float* belief156) {
  return Trap([&]() -> int {
    for (int k = 0; k < 3; ++k) {
        for (int c = 0; c < 52; ++c) {
            E.belief[k][c] = 1.0f / (1.0f + std::exp(-belief156[k * 52 + c]));
        }
    }
    E.root_pending = false;
    // softmax over the hand's logits -> pick probabilities
    size_t nl = E.legal.size();
    std::vector<double> p(nl);
    double mx = -1e30, tot = 0.0;
    for (size_t i = 0; i < nl; ++i) mx = std::max(mx, (double)logits52[E.legal[i]]);
    for (size_t i = 0; i < nl; ++i) {
        p[i] = std::exp((double)logits52[E.legal[i]] - mx);
        tot += p[i];
    }
    for (size_t i = 0; i < nl; ++i) p[i] = std::max(p[i] / tot, 1e-6);

    auto add_combo = [&](std::array<int, 3> c) {
        std::sort(c.begin(), c.end());
        if (std::find(E.combos.begin(), E.combos.end(), c) == E.combos.end())
            E.combos.push_back(c);
    };
    // top-3 by probability
    {
        std::vector<size_t> idx(nl);
        for (size_t i = 0; i < nl; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 3, idx.end(),
                          [&](size_t a, size_t b) { return p[a] > p[b]; });
        add_combo({E.legal[idx[0]], E.legal[idx[1]], E.legal[idx[2]]});
    }
    if (E.have_actual) add_combo(E.actual_combo);   // review anchors the real pass; live has none
    int tries = 0;
    while ((int)E.combos.size() < E.n_cand && tries++ < E.n_cand * 8) {
        std::array<int, 3> c{};
        std::vector<bool> taken(nl, false);
        for (int pick = 0; pick < 3; ++pick) {
            double t2 = 0.0;
            for (size_t i = 0; i < nl; ++i) if (!taken[i]) t2 += p[i];
            std::uniform_real_distribution<double> u(0.0, t2);
            double r = u(E.rng);
            size_t chosen = 0;
            for (size_t i = 0; i < nl; ++i) {
                if (taken[i]) continue;
                chosen = i;
                if (r < p[i]) break;
                r -= p[i];
            }
            taken[chosen] = true;
            c[pick] = E.legal[chosen];
        }
        add_combo(c);
    }
    // determinizations + scripted sims (skip-and-retry; see FillDets)
    E.FillDets(E.dets, E.K);
    for (size_t ci = 0; ci < E.combos.size(); ++ci) {
        for (const auto& det : E.dets) {
            Sim s(E.env.Clone(), (int)ci);
            s.env.ResetForPassSearch(det);
            s.script.assign(E.combos[ci].begin(), E.combos[ci].end());
            s.script_seat = E.me;
            E.sims.push_back(std::move(s));
        }
    }
    return E.NextRolloutRequest();
  });
}

KEEP int32_t* an_result_combo() { return E.r_combo.data(); }

// CARDS-ONLY MATCH: K replicas of the whole match over the SAME dealt
// hands with the policy playing every seat (the worker SAMPLES actions,
// which is where replicas diverge). Result per replica: final totals,
// finished flag, deals played - the fate written in the cards.
KEEP int an_cards_match(int k) {
  return Trap([&]() -> int {
    E.playout_mode = E.trace_mode = E.pass_mode = false;
    E.cards_mode = true;
    E.sims.clear();
    E.dets.clear();
    if (E.deal_hands.empty()) return -1;
    E.rng.seed(0x00C0FFEEu);
    E.cards_deal.assign(k, 0);
    for (int r = 0; r < k; ++r) {
        HeartsEnv e(E.match_seed, true);
        e.Reset();
        e.SetDeal(E.deal_hands[0]);
        E.sims.emplace_back(std::move(e), r);
    }
    E.root_pending = false;
    return E.NextRolloutRequest();
  });
}

KEEP int an_cards_n() { return (int)(E.r_cards.size() / 6); }
KEEP int32_t* an_result_cards() { return E.r_cards.data(); }

KEEP int an_analyze(int deal_idx, int action_idx, int k) {
  return Trap([&]() -> int {
    E.K = k;
    E.playout_mode = E.trace_mode = false;
    E.pass_mode = E.cards_mode = false;
    // CRN: a position's determinization draws depend ONLY on its
    // coordinates, never on what was analyzed before it - reruns are
    // bit-identical and fp16/fp32 comparisons share worlds.
    E.rng.seed(0x9E3779B9u ^ (unsigned)(deal_idx * 131071 + action_idx * 257 + k));
    E.sims.clear();
    E.dets.clear();
    if (!E.SeekTo(deal_idx, action_idx)) return -1;
    if (E.env.IsPassing()) return -1;
    E.legal = Engine::LegalVector(E.env);
    E.me = E.env.GetCurrentPlayer();
    if (E.legal.size() < 2) {   // forced: trivially done, counts=0 marks it
        E.r_actions.assign(E.legal.begin(), E.legal.end());
        E.r_mean.assign(E.legal.size(), 0.0f);
        E.r_se.assign(E.legal.size(), 0.0f);
        E.r_pts.assign(E.legal.size(), 0.0f);
        E.r_n.assign(E.legal.size(), 0);
        E.stage = PUMP_DONE;
        return PUMP_DONE;
    }
    E.BuildContext();
    return E.RequestRoot();
  });
}

KEEP int an_rows() {
    return E.stage == PUMP_EQUITY ? (int)E.eq_rows.size()
         : E.root_pending ? 1 : (int)E.active.size();
}
// Current policy request is the ROOT call (feed logits+belief via
// an_feed_root) vs a rollout round (feed acts via an_feed_acts).
KEEP int an_is_root() { return E.root_pending ? 1 : 0; }
KEEP float* an_obs() { return E.obs_buf.data(); }
KEEP uint8_t* an_mask() { return E.mask_buf.data(); }
KEEP float* an_eq_in() { return E.eq_buf.data(); }
KEEP int32_t* an_act_in() { return E.act_buf.data(); }
KEEP float* an_f_in() { return E.f_in_buf.data(); }

// Feed the ROOT forward's outputs (logits unused for play analysis, belief
// drives determinization weighting), then sims spawn and rollouts begin.
KEEP int an_feed_root(const float* belief156) {
  return Trap([&]() -> int {
    for (int k = 0; k < 3; ++k) {
        for (int c = 0; c < 52; ++c) {
            float v = belief156[k * 52 + c];
            E.belief[k][c] = 1.0f / (1.0f + std::exp(-v));   // sigmoid
        }
    }
    E.root_pending = false;
    E.SpawnSims();
    return E.NextRolloutRequest();
  });
}

// Feed rollout argmax actions for the current active set. Every action is
// validated against that sim's legal set: an illegal act (a broken or
// unsupported backend output path, e.g. int64 ArgMax on some WebGPU
// stacks) would otherwise step nothing and spin the rollout loop forever.
// Returns -2 so the JS side can fall back to logits + JS-side argmax.
KEEP int an_feed_acts() {
  return Trap([&]() -> int {
    for (size_t j = 0; j < E.active.size(); ++j) {
        Sim& s = E.sims[E.active[j]];
        int a = E.act_buf[j];
        auto lr = s.env.GetLegalActions();
        bool ok = false;
        for (int i = 0; i < 13; ++i) {
            if (lr[i] == a) ok = true;
        }
        if (!ok) return -2;
        s.done = s.env.Step(a).done;
        E.CardsAdvance(s);
    }
    return E.NextRolloutRequest();
  });
}

// Feed equity P(place 1) per pending row.
KEEP int an_feed_equity() {
  return Trap([&]() -> int {
    for (size_t j = 0; j < E.eq_rows.size(); ++j) {
        E.sims[E.eq_rows[j]].result = E.f_in_buf[j];
    }
    return E.Finish();
  });
}

// Debug/test: build a random-self-play match internally and load it
// (lets the JS boundary be exercised without real match data).
KEEP int an_debug_selfplay(unsigned seed) {
    std::mt19937 rr(seed);
    HeartsEnv env(seed, true);
    std::vector<int> offsets = {0};
    std::vector<int> actions, hands;
    std::array<double, 4> tot{};
    int deals = 0;
    while (deals < 60) {
        env.Reset();
        for (int s = 0; s < 4; ++s) {
            for (const auto& c : env.GetState().hands[s]) {
                hands.push_back((int)c.suit * 13 + (c.rank - 2));
            }
        }
        bool done = false;
        while (!done) {
            auto lr = env.GetLegalActions();
            std::vector<int> lg;
            for (int i = 0; i < 13; ++i) {
                if (lr[i] != -1) lg.push_back(lr[i]);
            }
            int a = lg[rr() % lg.size()];
            actions.push_back(a);
            done = env.Step(a).done;
        }
        offsets.push_back((int)actions.size());
        auto sc = env.GetRoundScores();
        for (int i = 0; i < 4; ++i) tot[i] += sc[i];
        deals++;
        if (*std::max_element(tot.begin(), tot.end()) >= 100.0) break;
    }
    return an_load_match(seed, offsets.data(), deals, actions.data(),
                         hands.data());
}

KEEP int an_result_n() { return (int)E.r_actions.size(); }
KEEP int32_t* an_result_actions() { return E.r_actions.data(); }
KEEP float* an_result_mean() { return E.r_mean.data(); }
KEEP float* an_result_se() { return E.r_se.data(); }
KEEP int32_t* an_result_counts() { return E.r_n.data(); }
KEEP float* an_result_pts() { return E.r_pts.data(); }

}  // extern "C"

#ifdef ANALYSIS_NATIVE_TEST
// Native smoke: random match self-play with a uniform-random "policy",
// exercising the full pump loop without any net (feeds random legal acts,
// equity 0.25). Validates state-machine plumbing, replay, sampling.
#include <cstdio>
int main() {
    // Build a match by random self-play, recording actions.
    std::mt19937 rr(7);
    HeartsEnv env(424242, true);
    std::vector<int> offsets = {0};
    std::vector<int> actions, hands;
    std::array<double, 4> tot{};
    int deals = 0;
    while (deals < 60) {
        env.Reset();
        for (int s = 0; s < 4; ++s) {
            for (const auto& c : env.GetState().hands[s]) {
                hands.push_back((int)c.suit * 13 + (c.rank - 2));
            }
        }
        bool done = false;
        while (!done) {
            auto lr = env.GetLegalActions();
            std::vector<int> legal;
            for (int i = 0; i < 13; ++i) {
                if (lr[i] != -1) legal.push_back(lr[i]);
            }
            int a = legal[rr() % legal.size()];
            actions.push_back(a);
            done = env.Step(a).done;
        }
        offsets.push_back((int)actions.size());
        auto sc = env.GetRoundScores();
        for (int i = 0; i < 4; ++i) tot[i] += sc[i];
        deals++;
        if (*std::max_element(tot.begin(), tot.end()) >= 100.0) break;
    }
    printf("built match: %d deals, %d actions\n", deals, (int)actions.size());

    an_init(1);
    an_load_match(424242, offsets.data(), deals, actions.data(), hands.data());
    int analyzed = 0, rounds = 0, forced_free = 0;
    std::mt19937 fr(9);
    for (int d = 0; d < deals; d += 2) {
        int n_act = offsets[d + 1] - offsets[d];
        int idx = n_act > 30 ? 20 : n_act / 2;
        int kind = an_analyze(d, idx, 8);
        if (kind < 0) continue;
        while (kind != 0) {
            int rows = an_rows();
            if (kind == 1) {
                if (an_is_root()) {
                    // root: feed zero belief (uniform weighting)
                    static float bel[156] = {};
                    kind = an_feed_root(bel);
                } else {
                    uint8_t* m = an_mask();
                    int32_t* a = an_act_in();
                    for (int r = 0; r < rows; ++r) {
                        std::vector<int> lg;
                        for (int c = 0; c < 52; ++c) {
                            if (m[r * 52 + c]) lg.push_back(c);
                        }
                        a[r] = lg[fr() % lg.size()];
                    }
                    rounds++;
                    kind = an_feed_acts();
                }
            } else {
                float* p = an_f_in();
                for (int r = 0; r < an_rows(); ++r) p[r] = 0.25f;
                kind = an_feed_equity();
            }
        }
        int n = an_result_n();
        int32_t* cn = an_result_counts();
        for (int i = 0; i < n; ++i) {
            if (cn[i] != 8 && n > 1) { printf("BAD count\n"); return 1; }
        }
        analyzed++;
    }
    (void)forced_free;
    printf("analyzed %d decisions, %d policy rounds - SMOKE OK\n",
           analyzed, rounds);
    return 0;
}
#endif
