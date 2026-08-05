#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "HeartsEnv.hpp"
#include "InferenceServer.hpp"

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Probe which observation width a traced module expects. The observation
// layout is prefix-stable across engine versions, so feeding an older model
// the prefix of the current observation is always valid.
inline int ProbeObsDim(torch::jit::script::Module& m) {
    // 550 MUST precede 556: a 550-traced v5 net is all fixed-index slices and
    // would silently accept 556 input (ignoring the tail), while a 556 match
    // trace errors on 550 input - so probing 550 first resolves both
    // correctly. (556 = 550 + MATCH_CTX_DIM, hearts_net.py.)
    for (int dim : {550, 556, 238, 181}) {
        try {
            std::vector<torch::jit::IValue> probe;
            probe.push_back(torch::zeros({1, dim}, torch::kFloat32));
            probe.push_back(torch::ones({1, 52}, torch::kBool));
            m.forward(probe);
            return dim;
        } catch (const std::exception&) {
            // try the next known width
        }
    }
    return 0;
}

// A plain argmax policy over a traced module (any generation).
class RawPolicy {
public:
    RawPolicy(torch::jit::script::Module model, int obs_dim)
        : model_(std::move(model)), obs_dim_(obs_dim) {}

    int ChooseAction(const HeartsEnv& env) {
        auto obs = env.Observe();
        auto legal_raw = env.GetLegalActions();
        torch::Tensor o = torch::from_blob((void*)obs.data(), {1, obs_dim_}, torch::kFloat32).clone();
        torch::Tensor m = torch::zeros({1, 52}, torch::kBool);
        bool* mp = m.data_ptr<bool>();
        for (int i = 0; i < 13; ++i) {
            if (legal_raw[i] != -1) mp[legal_raw[i]] = true;
        }
        torch::NoGradGuard g;
        auto out = model_.forward({o, m}).toTuple();
        return out->elements()[0].toTensor().argmax(1).item<int>();
    }

    int ObsDim() const { return obs_dim_; }

private:
    torch::jit::script::Module model_;
    int obs_dim_;
};

// Common interface for decision-time players (flat Monte Carlo and the tree
// search) so generators and evaluators can hold either interchangeably.
class IPlayer {
public:
    virtual ~IPlayer() = default;
    virtual int ChooseAction(const HeartsEnv& env) = 0;
    // Soft teacher target for the most recent decision
    virtual const std::array<float, 52>& LastPolicy() const = 0;
};

// ---------------------------------------------------------------------------
// Decision-time determinized search
// ---------------------------------------------------------------------------
// Play decisions: sample K complete hidden-hand hypotheses consistent with
// everything observed (hand sizes, observed voids, cards we passed), weighted
// by the network's belief head; then for each legal action, play it in every
// hypothesis and roll all seats out to the end of the round with the policy
// network. The same K determinizations are shared across actions, so the
// comparison between actions is paired.
//
// Passing decisions (optional, cfg.pass_search): passing is informationally
// simultaneous, so each candidate 3-card pass is evaluated from a REWOUND
// pass state (full 13-card determinized hands, all picks redone): our picks
// are scripted to the candidate, everyone else picks and plays by policy.
// Candidates are generated from the policy's own pass distribution.
class SearchPlayer : public IPlayer {
public:
    struct Config {
        int determinizations = 32;
        bool belief_weighted = true;
        unsigned int seed = 12345;
        bool pass_search = false;
        int pass_candidates = 12;
        int pass_k = 12;
        // Temperature (in points) for the soft teacher target over action
        // values: near-tie actions share mass instead of an arbitrary one-hot.
        float target_temp = 1.0f;
        // Rollout depth: -1 (or >=13) rolls every simulation to the end of
        // the round (classic behavior). N truncates a simulation once N
        // tricks have completed past its start and scores the leaf with the
        // VALUE HEAD from the searching seat's perspective - V(s) predicts
        // the final relative round score in the same units the terminal
        // rollout would produce.
        int rollout_tricks = -1;
        // Probe logging (docs/match_aware_search_design.md, SNR/flip-rate
        // gates): when non-empty, every probe_every-th play decision dumps
        // per-action x per-determinization completed-rollout DEAL SCORES plus
        // the harness-provided match context. LOGGING ONLY - no scoring
        // change. Requires full rollouts (rows skipped under truncation).
        std::string probe_log;
        int probe_every = 5;
        // Adaptive-K probe (2026-07-25): when >0 and the harness-provided
        // match context shows an endgame state (max total >= 85), use this
        // many determinizations instead of `determinizations`. SNR scales
        // ~sqrt(K); endgame states are ~16% of decisions so the cost is
        // ~1.5x, not 4x.
        int k_endgame = 0;
        // MATCH-AWARE MODE (docs/match_aware_search_design.md): the model is
        // a 556-dim trace; every net call carries the ACTING seat's match
        // context (silent-null rule: never score one seat's decision from
        // another's perspective), and completed rollouts are scored by the
        // equity model (win objective: P(place 1) incl. tie mass) instead of
        // deal points. Requires FULL rollouts.
        bool match_aware = false;
        std::shared_ptr<torch::jit::script::Module> equity_model;
        // Evaluate truncated-rollout leaves with the ORACLE head (which sees
        // the determinized hands) instead of the visible-info value head.
        // Requires a trace exposing the "oracle" method; measured 2026-07-14
        // that visible-info leaves collapse because they cannot distinguish
        // the K determinized worlds.
        bool oracle_leaves = false;
        // Inference device (used by the module-owning constructor, which
        // wraps the model in a DirectBackend). jit::Module copies share
        // storage, so moving the model moves it for every SearchPlayer built
        // from the same loaded module - probe obs width BEFORE constructing
        // players.
        torch::Device device = torch::kCPU;
        // bf16 inference on CUDA (~1.5-2x): safe for search - logits feed
        // argmax/sampling and values are averaged over many rollouts
        bool bf16 = false;
        // Optional separate net for belief marginals (determinization
        // sampling only): lets a strong rollout policy borrow a
        // better-calibrated belief head from another checkpoint. Null =
        // use the main backend.
        std::shared_ptr<InferenceBackend> belief_backend;
        // Moon-shooter mode (exploiter league round 1,
        // docs/exploiter_league_prereg.md): "" (off), "agg", or "sel".
        // While the moon is alive for this seat, candidate moves are
        // scored by MOON PROBABILITY - the fraction of completed rollouts
        // (this seat playing a moon-seeking heuristic inside the rollout,
        // everyone else the policy) where this seat takes all 26. AGG
        // plays the moon argmax from the pass until the moon is
        // mathematically dead; SEL additionally estimates the shoot
        // line's MATCH EQUITY vs the normal line's at every alive
        // decision and shoots only while shoot >= normal. Requires
        // match_aware (equity units + completed-rollout scoring).
        std::string shooter_mode;
    };

    // Shared-backend constructor: many players (threads) funnel inference
    // through one backend, e.g. a ServedBackend on an InferenceServer.
    // (Default-Config overloads instead of `Config cfg = Config()` default
    // arguments: GCC rejects an NSDMI class as a default arg inside the
    // enclosing class; MSVC accepted it silently.)
    SearchPlayer(std::shared_ptr<InferenceBackend> backend, int model_obs_dim,
                 Config cfg)
        : backend_(std::move(backend)), obs_dim_(model_obs_dim), cfg_(cfg), rng_(cfg.seed) {
        if (cfg_.match_aware) {
            if (obs_dim_ != 556)
                throw std::runtime_error("match_aware requires a 556-dim trace");
            if (!cfg_.equity_model)
                throw std::runtime_error("match_aware requires an equity model");
            if (cfg_.rollout_tricks >= 0)
                throw std::runtime_error("match_aware requires FULL rollouts "
                                         "(equity units != value-head units)");
        }
        if (!cfg_.shooter_mode.empty()) {
            if (cfg_.shooter_mode != "agg" && cfg_.shooter_mode != "sel")
                throw std::runtime_error("shooter_mode must be \"agg\" or \"sel\"");
            if (!cfg_.match_aware)
                throw std::runtime_error("shooter_mode requires match_aware");
        }
    }

    SearchPlayer(std::shared_ptr<InferenceBackend> backend, int model_obs_dim)
        : SearchPlayer(std::move(backend), model_obs_dim, Config()) {}

    SearchPlayer(torch::jit::script::Module model, int model_obs_dim, Config cfg)
        : SearchPlayer(std::make_shared<DirectBackend>(std::move(model), cfg.device, cfg.bf16),
                       model_obs_dim, cfg) {}

    SearchPlayer(torch::jit::script::Module model, int model_obs_dim)
        : SearchPlayer(std::move(model), model_obs_dim, Config()) {}

    // Per-action search statistics for the LAST ChooseAction call (record
    // format v2, docs/expert_iter_v2_prereg.md). Populated only for searched
    // PLAY decisions (valid == true): pass picks and forced single-action
    // moves carry zeroed stats with pass_phase / valid marking why. Filled
    // from quantities the search already computes - no behavior change.
    struct ActionStats {
        std::vector<int> actions;   // legal action ids, search order
        std::vector<float> mean;    // per-action mean rollout value
        std::vector<float> se;      // std of per-det results / sqrt(n); 0 if n<2
        std::vector<int> n;         // per-action determinization count
        int n_dets = 0;             // dets used for this decision (k vs k_endgame)
        bool pass_phase = false;    // decision was a passing-phase pick
        bool valid = false;         // per-action stats populated
    };
    const ActionStats& LastStats() const { return last_stats_; }

    // Shooter-mode state for the LAST ChooseAction call (Phase A logging).
    // valid only when cfg_.shooter_mode is set. alive = the moon was still
    // possible for this seat at the decision; shooting = the decision was
    // made under moon scoring (AGG: always while alive; SEL: only when the
    // shoot line's equity won). The pass commit decision is made once, on
    // the FIRST pick; the 2nd/3rd queued picks report shooting=false, so
    // read the commit state from the first pick only.
    struct ShooterStats {
        bool valid = false;
        bool alive = false;
        bool shooting = false;
        bool pass_phase = false;
        float moon_p = 0.0f;    // moon prob of the chosen shoot action
        float eq_shoot = 0.0f;  // match equity of the shoot line
        float eq_norm = 0.0f;   // SEL only: match equity of the normal line
    };
    const ShooterStats& LastShooter() const { return last_shooter_; }

    int ChooseAction(const HeartsEnv& env) override {
        std::vector<int> legal = LegalVector(env);
        if (!cfg_.shooter_mode.empty()) {
            last_shooter_ = ShooterStats();
            last_shooter_.valid = true;
            last_shooter_.pass_phase = env.IsPassing();
            last_shooter_.alive = MoonAliveFor(env, env.GetCurrentPlayer());
        }
        if (env.IsPassing()) {
            last_stats_ = ActionStats();
            last_stats_.pass_phase = true;
            return SetOneHot(ChoosePass(env, legal));
        }
        if (legal.size() == 1) {
            last_stats_ = ActionStats();
            return SetOneHot(legal[0]);
        }

        int me = env.GetCurrentPlayer();
        if (!cfg_.shooter_mode.empty() && last_shooter_.alive) {
            // Moon decision: same K schedule as the normal search
            int nd = cfg_.determinizations;
            if (cfg_.k_endgame > 0 &&
                *std::max_element(probe_totals_.begin(), probe_totals_.end()) >= 85.0) {
                nd = cfg_.k_endgame;
            }
            last_stats_ = ActionStats();   // shooter decisions carry no v2 stats
            return SetOneHot(ChooseShoot(env, legal, nd));
        }

        // Sample K determinizations, shared across candidate actions
        // (K boosted in endgame score states when k_endgame is set)
        int n_dets = cfg_.determinizations;
        if (cfg_.k_endgame > 0 &&
            *std::max_element(probe_totals_.begin(), probe_totals_.end()) >= 85.0) {
            n_dets = cfg_.k_endgame;
        }
        BuildContext(env);
        std::vector<std::array<std::vector<int>, 4>> dets(n_dets);
        for (auto& d : dets) d = SampleHands(env);

        std::vector<Sim> sims;
        sims.reserve(legal.size() * dets.size());
        int start_tricks = env.GetState().tricks_played;
        for (size_t ai = 0; ai < legal.size(); ++ai) {
            for (const auto& det : dets) {
                Sim s(env.Clone(), static_cast<int>(ai));
                s.eval_seat = me;
                s.start_tricks = start_tricks;
                s.sim_env.SetHands(det);
                s.done = s.sim_env.Step(legal[ai]).done;
                sims.push_back(std::move(s));
            }
        }
        RolloutAll(sims);

        if (!cfg_.probe_log.empty() && cfg_.rollout_tricks < 0
            && (++probe_decisions_ % cfg_.probe_every == 0)) {
            if (!probe_out_.is_open()) {
                probe_out_.open(cfg_.probe_log);
                probe_out_ << "decision,seat,deals_played,t0,t1,t2,t3,"
                              "n_actions,action_card,det,s0,s1,s2,s3\n";
            }
            const size_t K = dets.size();
            for (size_t i = 0; i < sims.size(); ++i) {
                const auto& s = sims[i];
                if (!s.done) continue;
                auto sc = s.sim_env.GetRoundScores();
                probe_out_ << probe_decisions_ << ',' << me << ','
                           << probe_deals_ << ',' << probe_totals_[0] << ','
                           << probe_totals_[1] << ',' << probe_totals_[2] << ','
                           << probe_totals_[3] << ',' << legal.size() << ','
                           << legal[s.tag] << ',' << (i % K) << ',' << sc[0]
                           << ',' << sc[1] << ',' << sc[2] << ',' << sc[3] << '\n';
            }
            probe_out_.flush();
        }

        std::vector<double> score(legal.size(), 0.0);
        std::vector<double> sumsq(legal.size(), 0.0);
        std::vector<int> cnt(legal.size(), 0);
        for (const auto& s : sims) {
            score[s.tag] += s.result;
            sumsq[s.tag] += s.result * s.result;
            cnt[s.tag] += 1;
        }
        size_t best = 0;
        for (size_t ai = 1; ai < legal.size(); ++ai) {
            if (score[ai] > score[best]) best = ai;
        }

        // Per-action stats for the v2 record writer (mean, SE of the mean).
        last_stats_ = ActionStats();
        last_stats_.valid = true;
        last_stats_.n_dets = n_dets;
        last_stats_.actions = legal;
        last_stats_.n = cnt;
        last_stats_.mean.resize(legal.size());
        last_stats_.se.resize(legal.size());
        for (size_t ai = 0; ai < legal.size(); ++ai) {
            int nc = cnt[ai];
            double mu = nc > 0 ? score[ai] / nc : 0.0;
            last_stats_.mean[ai] = static_cast<float>(mu);
            double se = 0.0;
            if (nc >= 2) {
                double var = (sumsq[ai] - nc * mu * mu) / (nc - 1);
                if (var < 0.0) var = 0.0;   // numerical guard
                se = std::sqrt(var / nc);
            }
            last_stats_.se[ai] = static_cast<float>(se);
        }

        // Soft teacher target: softmax over mean action values (points).
        last_pi_.fill(0.0f);
        double denom = 0.0;
        std::vector<double> ex(legal.size());
        for (size_t ai = 0; ai < legal.size(); ++ai) {
            double gap = (score[ai] - score[best]) / static_cast<double>(dets.size());
            ex[ai] = std::exp(gap / cfg_.target_temp);
            denom += ex[ai];
        }
        for (size_t ai = 0; ai < legal.size(); ++ai) {
            last_pi_[legal[ai]] = static_cast<float>(ex[ai] / denom);
        }
        return legal[best];
    }

    // Teacher target distribution for the most recent ChooseAction call.
    // Play decisions: softmax over per-action mean search values; pass picks
    // and forced moves: one-hot.
    const std::array<float, 52>& LastPolicy() const override { return last_pi_; }

    // Probe-log context: the match harness updates carried totals/deals so
    // logged decisions carry the score state (SearchPlayer itself remains
    // match-blind - this feeds the OFFLINE flip-rate/SNR analysis only).
    void SetMatchContext(const std::array<double, 4>& totals, int deals) {
        probe_totals_ = totals;
        probe_deals_ = deals;
    }

    // Sample one determinization for the current context (public: selftest).
    // BuildContext(env) must have been called for this state first.
    std::array<std::vector<int>, 4> SampleHands(const HeartsEnv& env) {
        for (int attempt = 0; attempt < 200; ++attempt) {
            bool use_belief = ctx_.have_belief && cfg_.belief_weighted && attempt < 100;
            std::array<std::vector<int>, 4> hands;
            if (TrySample(use_belief, hands)) return hands;
        }
        // The randomized greedy can wedge on tightly-constrained endgame
        // states even though an assignment always exists (the true hands are
        // one). The exact solver cannot fail on satisfiable constraints.
        std::array<std::vector<int>, 4> hands;
        if (TrySampleExact(hands)) return hands;
        throw std::runtime_error("SampleHands: constraints unsatisfiable (engine state bug)");
    }

    void BuildContext(const HeartsEnv& env) {
        ctx_ = Context();
        ctx_.me = env.GetCurrentPlayer();
        const auto& played_by = env.GetPlayedBy();
        const auto& voids = env.GetVoidTracker();

        std::array<bool, 52> mine{};
        for (const auto& c : env.GetState().hands[ctx_.me]) {
            int id = (static_cast<int>(c.suit) * 13) + (c.rank - 2);
            mine[id] = true;
            ctx_.my_hand.push_back(id);
        }
        for (int c = 0; c < 52; ++c) {
            bool played = played_by[0][c] || played_by[1][c] || played_by[2][c] || played_by[3][c];
            if (!played && !mine[c]) ctx_.unseen.push_back(c);
        }
        for (int k = 1; k < 4; ++k) {
            int abs_seat = (ctx_.me + k) % 4;
            ctx_.cap[k - 1] = env.GetHandSize(abs_seat);
            for (int s = 0; s < 4; ++s) {
                ctx_.is_void[k - 1][s] = voids[abs_seat * 4 + s];
            }
        }
        // Cards we passed are pinned to the receiver while unplayed
        ctx_.pinned_rel.assign(52, -1);
        if (env.GetPassDirection() != 3) {
            int offset = (env.GetPassDirection() == 0) ? 1 : (env.GetPassDirection() == 1) ? 3 : 2;
            for (int a : env.GetPassPicks(ctx_.me)) {
                bool unseen = std::find(ctx_.unseen.begin(), ctx_.unseen.end(), a) != ctx_.unseen.end();
                if (unseen) ctx_.pinned_rel[a] = offset - 1;
            }
        }
        // Belief marginals from the network (needs the 3-output search trace)
        ctx_.have_belief = FetchBelief(env);
    }

private:
    struct Context {
        int me = 0;
        std::vector<int> my_hand;
        std::vector<int> unseen;
        std::array<int, 3> cap{};
        std::array<std::array<bool, 4>, 3> is_void{};
        std::vector<int> pinned_rel;   // card -> relative opponent (0..2) or -1
        float belief[3][52] = {};
        bool have_belief = false;
    };

    struct Sim {
        HeartsEnv sim_env;
        int tag;
        bool done = false;
        bool truncated = false;   // stopped early; leaf goes to the value head
        int eval_seat = 0;        // the searching seat this sim scores for
        int start_tricks = 0;     // tricks_played when the sim was spawned
        double result = 0.0;      // filled by RolloutAll
        // Scripted pass picks: while sim is passing and it's script_seat's
        // turn, play from script instead of the policy
        std::vector<int> script;
        int script_seat = -1;
        size_t script_pos = 0;
        // Moon rollout continuation: while the moon is alive for this seat,
        // it plays MoonHeuristicAction inside the rollout instead of the
        // policy (reverts to policy once the moon dies). -1 = off.
        int moon_seat = -1;

        Sim(HeartsEnv e, int t) : sim_env(std::move(e)), tag(t) {}
    };

    static double RelReward(const HeartsEnv& env, int seat) {
        auto sc = env.GetRoundScores();
        double avg = (sc[0] + sc[1] + sc[2] + sc[3]) / 4.0;
        return avg - sc[seat];
    }

    // ------------------------- moon-shooter helpers -------------------------

    // Mid-deal round_scores are RAW accumulated points (the 0/26/26/26 moon
    // adjustment happens only at trick 13), so: alive iff no OTHER seat has
    // taken a point this deal.
    static bool MoonAliveFor(const HeartsEnv& env, int seat) {
        auto sc = env.GetRoundScores();
        for (int i = 0; i < 4; ++i) {
            if (i != seat && sc[i] > 0) return false;
        }
        return true;
    }

    // Completed-deal moon signature: post-adjustment scores are 0/26/26/26
    // (total 78) with the shooter at 0.
    static bool MoonSuccess(const HeartsEnv& env, int seat) {
        auto sc = env.GetRoundScores();
        return (sc[0] + sc[1] + sc[2] + sc[3]) == 78 && sc[seat] == 0;
    }

    // Rollout continuation for a committed shooter: cash winners, keep
    // control cards. Deliberately simple - it only needs to be
    // moon-DIRECTED, and the same policy applies to every candidate action
    // over the same shared determinizations, so its biases pair out of the
    // per-action comparison.
    static int MoonHeuristicAction(const HeartsEnv& env) {
        std::vector<int> legal;
        {
            auto lr = env.GetLegalActions();
            for (int i = 0; i < 13; ++i) {
                if (lr[i] != -1) legal.push_back(lr[i]);
            }
        }
        if (legal.size() == 1) return legal[0];
        const auto& st = env.GetState();
        const auto& played_by = env.GetPlayedBy();
        int me = env.GetCurrentPlayer();
        std::array<bool, 52> mine{};
        for (const auto& c : st.hands[me]) {
            mine[static_cast<int>(c.suit) * 13 + c.rank - 2] = true;
        }
        auto gone = [&](int id) {
            return mine[id] || played_by[0][id] || played_by[1][id]
                   || played_by[2][id] || played_by[3][id];
        };
        auto is_master = [&](int id) {   // no higher card of its suit is live
            for (int r = id % 13 + 1; r < 13; ++r) {
                if (!gone((id / 13) * 13 + r)) return false;
            }
            return true;
        };
        if (st.current_trick.empty()) {
            // Lead: highest master if any, else highest of the longest suit.
            int best = -1;
            for (int a : legal) {
                if (is_master(a) && (best < 0 || a % 13 > best % 13)) best = a;
            }
            if (best >= 0) return best;
            std::array<int, 4> cnt{};
            for (int a : legal) cnt[a / 13]++;
            int suit = 0;
            for (int s = 1; s < 4; ++s) {
                if (cnt[s] > cnt[suit]) suit = s;
            }
            for (int a : legal) {
                if (a / 13 == suit && (best < 0 || a % 13 > best % 13)) best = a;
            }
            return best;
        }
        int led = static_cast<int>(st.getLedSuit());
        bool have_led = false;
        for (int a : legal) {
            if (a / 13 == led) have_led = true;
        }
        if (have_led) {
            // Beat the current winner with our highest, else duck lowest.
            int win_rank = -1;
            for (const auto& pc : st.current_trick) {
                if (static_cast<int>(pc.card.suit) == led && pc.card.rank - 2 > win_rank)
                    win_rank = pc.card.rank - 2;
            }
            int hi = -1, lo = -1;
            for (int a : legal) {
                if (a / 13 != led) continue;
                if (hi < 0 || a % 13 > hi % 13) hi = a;
                if (lo < 0 || a % 13 < lo % 13) lo = a;
            }
            return (hi % 13 > win_rank) ? hi : lo;
        }
        // Void in the led suit: discard the lowest non-point card (hearts and
        // the QS are the shooter's ammunition), else the lowest point card.
        int best = -1;
        for (int a : legal) {
            if (a / 13 == 3 || a == 36) continue;   // hearts / QS
            if (best < 0 || a % 13 < best % 13) best = a;
        }
        if (best >= 0) return best;
        for (int a : legal) {
            if (best < 0 || a % 13 < best % 13) best = a;
        }
        return best;
    }

    // Moon pass candidates the policy distribution would not propose: the
    // shooter KEEPS controls and dumps its lowest cards. Two heuristics:
    // 3 lowest overall, 3 lowest non-hearts (hearts must be won back).
    static void AddMoonPassCandidates(const std::vector<int>& legal,
                                      std::vector<std::array<int, 3>>& combos) {
        auto add = [&](std::array<int, 3> c) {
            std::sort(c.begin(), c.end());
            if (std::find(combos.begin(), combos.end(), c) == combos.end())
                combos.push_back(c);
        };
        std::vector<int> by_rank = legal;
        std::sort(by_rank.begin(), by_rank.end(),
                  [](int a, int b) { return a % 13 < b % 13; });
        add({by_rank[0], by_rank[1], by_rank[2]});
        std::vector<int> nh;
        for (int a : by_rank) {
            if (a / 13 != 3) nh.push_back(a);
        }
        if (nh.size() >= 3) add({nh[0], nh[1], nh[2]});
    }

    // Shooter play decision while the moon is alive: score legal moves by
    // moon probability over shared determinizations (this seat playing the
    // moon heuristic inside rollouts). SEL also runs the normal search on
    // the SAME determinizations and shoots only while the shoot line's
    // match equity >= the normal line's. Ties in moon probability break
    // toward higher equity (early decisions often tie at 0).
    int ChooseShoot(const HeartsEnv& env, const std::vector<int>& legal, int n_dets) {
        int me = env.GetCurrentPlayer();
        BuildContext(env);
        std::vector<std::array<std::vector<int>, 4>> dets(n_dets);
        for (auto& d : dets) d = SampleHands(env);
        int start_tricks = env.GetState().tricks_played;

        auto run = [&](bool moon, std::vector<double>& eq, std::vector<double>& mp) {
            std::vector<Sim> sims;
            sims.reserve(legal.size() * dets.size());
            for (size_t ai = 0; ai < legal.size(); ++ai) {
                for (const auto& det : dets) {
                    Sim s(env.Clone(), static_cast<int>(ai));
                    s.eval_seat = me;
                    s.start_tricks = start_tricks;
                    if (moon) s.moon_seat = me;
                    s.sim_env.SetHands(det);
                    s.done = s.sim_env.Step(legal[ai]).done;
                    sims.push_back(std::move(s));
                }
            }
            RolloutAll(sims);
            eq.assign(legal.size(), 0.0);
            mp.assign(legal.size(), 0.0);
            std::vector<int> n(legal.size(), 0);
            for (const auto& s : sims) {
                eq[s.tag] += s.result;
                mp[s.tag] += MoonSuccess(s.sim_env, me) ? 1.0 : 0.0;
                n[s.tag] += 1;
            }
            for (size_t ai = 0; ai < legal.size(); ++ai) {
                if (n[ai]) { eq[ai] /= n[ai]; mp[ai] /= n[ai]; }
            }
        };

        std::vector<double> meq, mmp;
        run(true, meq, mmp);
        size_t bm = 0;
        for (size_t ai = 1; ai < legal.size(); ++ai) {
            if (mmp[ai] > mmp[bm] || (mmp[ai] == mmp[bm] && meq[ai] > meq[bm])) bm = ai;
        }
        last_shooter_.moon_p = static_cast<float>(mmp[bm]);
        last_shooter_.eq_shoot = static_cast<float>(meq[bm]);
        if (cfg_.shooter_mode == "agg") {
            last_shooter_.shooting = true;
            return legal[bm];
        }
        std::vector<double> neq, nmp;
        run(false, neq, nmp);
        size_t bn = 0;
        for (size_t ai = 1; ai < legal.size(); ++ai) {
            if (neq[ai] > neq[bn]) bn = ai;
        }
        last_shooter_.eq_norm = static_cast<float>(neq[bn]);
        if (meq[bm] >= neq[bn]) {
            last_shooter_.shooting = true;
            return legal[bm];
        }
        return legal[bn];
    }

    // Match-aware helpers ---------------------------------------------------

    // Write the 6 match-context floats (indices 550..555) for `seat`.
    void WriteCtx(float* row, int seat) const {
        double mx = *std::max_element(probe_totals_.begin(), probe_totals_.end());
        for (int i = 0; i < 4; ++i) {
            row[550 + i] = static_cast<float>(probe_totals_[(seat + i) % 4] / 100.0);
        }
        row[554] = static_cast<float>(probe_deals_ / 20.0);
        row[555] = static_cast<float>((100.0 - mx) / 100.0);
    }

    // Copy an engine observation (550 floats) into a row of width obs_dim_,
    // appending ctx for `seat` when running the 556-dim match trace.
    template <typename Obs>
    void FillObsRow(float* row, const Obs& obs, int seat) const {
        std::memcpy(row, obs.data(), 550 * sizeof(float));
        if (obs_dim_ == 556) WriteCtx(row, seat);
    }

    // Score completed rollouts by match equity: exact placements when the
    // match ends this deal, else the equity net's P(place 1) for the
    // evaluating seat at the post-deal score state.
    void ScoreEquity(std::vector<Sim>& sims, const std::vector<size_t>& idx) {
        std::vector<size_t> live;
        std::vector<std::array<double, 4>> after(idx.size());
        for (size_t j = 0; j < idx.size(); ++j) {
            auto sc = sims[idx[j]].sim_env.GetRoundScores();
            auto t = probe_totals_;
            for (int i = 0; i < 4; ++i) t[i] += sc[i];
            after[j] = t;
            if (*std::max_element(t.begin(), t.end()) >= 100.0) {
                sims[idx[j]].result = TerminalWinValue(t, sims[idx[j]].eval_seat);
            } else {
                live.push_back(j);
            }
        }
        if (live.empty()) return;
        const int deals_after = probe_deals_ + 1;
        torch::Tensor x = torch::zeros({(long)live.size(), 10}, torch::kFloat32);
        float* xp = x.data_ptr<float>();
        for (size_t k = 0; k < live.size(); ++k) {
            const auto& t = after[live[k]];
            int seat = sims[idx[live[k]]].eval_seat;
            double mx = *std::max_element(t.begin(), t.end());
            float* row = xp + k * 10;
            for (int i = 0; i < 4; ++i) row[i] = (float)(t[(seat + i) % 4] / 100.0);
            row[4] = (float)(deals_after / 20.0);
            row[5] = (float)((100.0 - mx) / 100.0);
            row[6 + (deals_after % 4)] = 1.0f;
        }
        torch::NoGradGuard g;
        auto probs = torch::softmax(
            cfg_.equity_model->forward({x}).toTensor(), 1);
        auto acc = probs.accessor<float, 2>();
        for (size_t k = 0; k < live.size(); ++k) {
            sims[idx[live[k]]].result = acc[k][0];  // P(place 1)
        }
    }

    // Exact win value (P1 with tie mass) from final totals for `seat`.
    static double TerminalWinValue(const std::array<double, 4>& totals, int seat) {
        double mine = totals[seat];
        int better = 0, tied = 0;
        for (int i = 0; i < 4; ++i) {
            if (i == seat) continue;
            if (totals[i] < mine) ++better;
            else if (totals[i] == mine) ++tied;
        }
        if (better > 0) return 0.0;
        return 1.0 / (1 + tied);
    }

    std::vector<int> LegalVector(const HeartsEnv& env) {
        std::vector<int> legal;
        auto lr = env.GetLegalActions();
        for (int i = 0; i < 13; ++i) {
            if (lr[i] != -1) legal.push_back(lr[i]);
        }
        return legal;
    }

    // Roll every sim forward: scripted steps play for free, everything else
    // is batched policy argmax. A sim ends either at the round's end
    // (result = true relative reward) or, with cfg_.rollout_tricks >= 0, at
    // a trick boundary rollout_tricks past its start (result = value head,
    // asked from the searching seat's perspective, in the same units).
    void RolloutAll(std::vector<Sim>& sims) {
        const int T = cfg_.rollout_tricks;
        const bool truncate = (T >= 0 && T < 13);
        auto check_truncate = [&](Sim& s) {
            if (truncate && !s.done && !s.truncated && !s.sim_env.IsPassing()
                && s.sim_env.GetState().tricks_played - s.start_tricks >= T) {
                s.truncated = true;
            }
        };
        for (auto& s : sims) check_truncate(s);

        std::vector<size_t> active;
        while (true) {
            // Consume any scripted picks and moon-heuristic steps first
            // (no inference needed for either)
            bool progressed = true;
            while (progressed) {
                progressed = false;
                for (auto& s : sims) {
                    if (!s.done && s.script_pos < s.script.size() && s.sim_env.IsPassing()
                        && s.sim_env.GetCurrentPlayer() == s.script_seat) {
                        s.done = s.sim_env.Step(s.script[s.script_pos++]).done;
                        progressed = true;
                    }
                    if (!s.done && !s.truncated && s.moon_seat >= 0
                        && !s.sim_env.IsPassing()
                        && s.sim_env.GetCurrentPlayer() == s.moon_seat
                        && MoonAliveFor(s.sim_env, s.moon_seat)) {
                        s.done = s.sim_env.Step(MoonHeuristicAction(s.sim_env)).done;
                        check_truncate(s);
                        progressed = true;
                    }
                }
            }
            active.clear();
            for (size_t i = 0; i < sims.size(); ++i) {
                if (!sims[i].done && !sims[i].truncated) active.push_back(i);
            }
            if (active.empty()) break;

            torch::Tensor o = torch::empty({(long)active.size(), obs_dim_}, torch::kFloat32);
            torch::Tensor m = torch::zeros({(long)active.size(), 52}, torch::kBool);
            float* op = o.data_ptr<float>();
            bool* mp = m.data_ptr<bool>();
            for (size_t j = 0; j < active.size(); ++j) {
                auto obs = sims[active[j]].sim_env.Observe();
                FillObsRow(op + j * obs_dim_, obs,
                           sims[active[j]].sim_env.GetCurrentPlayer());
                auto lr = sims[active[j]].sim_env.GetLegalActions();
                for (int i = 0; i < 13; ++i) {
                    if (lr[i] != -1) mp[j * 52 + lr[i]] = true;
                }
            }
            torch::Tensor acts = backend_->Forward(o, m).logits.argmax(1);
            auto acc = acts.accessor<int64_t, 1>();
            for (size_t j = 0; j < active.size(); ++j) {
                Sim& s = sims[active[j]];
                s.done = s.sim_env.Step(static_cast<int>(acc[j])).done;
                check_truncate(s);
            }
        }

        // Resolve results: terminal sims score exactly; truncated leaves are
        // batched through the value head in one forward. In match-aware
        // mode completed rollouts are scored by MATCH EQUITY instead of
        // deal points (win objective).
        std::vector<size_t> leaves;
        std::vector<size_t> eq_batch;
        for (size_t i = 0; i < sims.size(); ++i) {
            if (sims[i].done) {
                if (cfg_.match_aware) {
                    eq_batch.push_back(i);
                } else {
                    sims[i].result = RelReward(sims[i].sim_env, sims[i].eval_seat);
                }
            } else {
                leaves.push_back(i);
            }
        }
        if (!eq_batch.empty()) ScoreEquity(sims, eq_batch);
        if (leaves.empty()) return;

        torch::Tensor o = torch::empty({(long)leaves.size(), obs_dim_}, torch::kFloat32);
        float* op = o.data_ptr<float>();
        for (size_t j = 0; j < leaves.size(); ++j) {
            Sim& s = sims[leaves[j]];
            auto obs = s.sim_env.ObserveFor(s.eval_seat);
            FillObsRow(op + j * obs_dim_, obs, s.eval_seat);
        }

        torch::Tensor v;
        if (cfg_.oracle_leaves && backend_->HasOracle()) {
            // The oracle sees the sim's remaining DETERMINIZED hands - the
            // information a visible-info value head provably lacks
            torch::Tensor h = torch::zeros({(long)leaves.size(), 156}, torch::kFloat32);
            float* hp = h.data_ptr<float>();
            for (size_t j = 0; j < leaves.size(); ++j) {
                Sim& s = sims[leaves[j]];
                const auto& hands = s.sim_env.GetState().hands;
                for (int k = 1; k < 4; ++k) {
                    int seat = (s.eval_seat + k) % 4;
                    for (const auto& c : hands[seat]) {
                        int id = static_cast<int>(c.suit) * 13 + (c.rank - 2);
                        hp[j * 156 + (k - 1) * 52 + id] = 1.0f;
                    }
                }
            }
            v = backend_->OracleForward(o, h);
        } else {
            torch::Tensor m = torch::ones({(long)leaves.size(), 52}, torch::kBool);
            v = backend_->Forward(o, m).value;
        }
        auto vacc = v.accessor<float, 2>();
        for (size_t j = 0; j < leaves.size(); ++j) {
            sims[leaves[j]].result = vacc[j][0];
        }
    }

    // ------------------------- passing-phase search -------------------------

    int ChoosePass(const HeartsEnv& env, const std::vector<int>& legal) {
        int me = env.GetCurrentPlayer();
        auto& queued = pending_pass_[me];
        if (!queued.empty()) {
            int a = queued.front();
            queued.erase(queued.begin());
            return a;
        }
        if (!cfg_.pass_search || legal.size() != 13) {
            return ArgmaxSingle(env, legal);
        }

        // Candidate 3-card combos from the policy's own pass distribution
        std::vector<float> probs = PolicyProbs(env, legal);
        std::set<std::array<int, 3>> combo_set;
        combo_set.insert(TopThree(legal, probs));
        int tries = 0;
        while ((int)combo_set.size() < cfg_.pass_candidates && tries++ < cfg_.pass_candidates * 8) {
            combo_set.insert(SampleCombo(legal, probs));
        }
        std::vector<std::array<int, 3>> combos(combo_set.begin(), combo_set.end());

        // Determinizations of the full deal (everyone back to 13 cards)
        BuildContext(env);
        ctx_.cap = {13, 13, 13};  // opponents' full hands, regardless of picks in flight
        std::vector<std::array<std::vector<int>, 4>> dets(cfg_.pass_k);
        for (auto& d : dets) d = SampleHands(env);

        // Evaluate a candidate list over the shared determinizations: mean
        // result (match equity when match_aware) and mean moon probability.
        auto eval = [&](const std::vector<std::array<int, 3>>& cs, bool moon,
                        std::vector<double>& eq, std::vector<double>& mp) {
            std::vector<Sim> sims;
            sims.reserve(cs.size() * dets.size());
            for (size_t ci = 0; ci < cs.size(); ++ci) {
                for (const auto& det : dets) {
                    Sim s(env.Clone(), static_cast<int>(ci));
                    s.sim_env.ResetForPassSearch(det);
                    s.eval_seat = me;
                    s.start_tricks = 0;  // pass search rewinds to the deal start
                    s.script.assign(cs[ci].begin(), cs[ci].end());
                    s.script_seat = me;
                    if (moon) s.moon_seat = me;
                    sims.push_back(std::move(s));
                }
            }
            RolloutAll(sims);
            eq.assign(cs.size(), 0.0);
            mp.assign(cs.size(), 0.0);
            std::vector<int> n(cs.size(), 0);
            for (const auto& s : sims) {
                eq[s.tag] += s.result;
                mp[s.tag] += MoonSuccess(s.sim_env, me) ? 1.0 : 0.0;
                n[s.tag] += 1;
            }
            for (size_t ci = 0; ci < cs.size(); ++ci) {
                if (n[ci]) { eq[ci] /= n[ci]; mp[ci] /= n[ci]; }
            }
        };

        if (!cfg_.shooter_mode.empty()) {
            // Moon pass (the moon is trivially alive at the pass). AGG
            // commits unconditionally; SEL prices the best moon pass against
            // the best normal pass in match equity and commits only if the
            // shoot line wins. The commit decision is made ONCE here; picks
            // 2/3 are queued from the chosen combo.
            std::vector<std::array<int, 3>> mcombos = combos;
            AddMoonPassCandidates(legal, mcombos);
            std::vector<double> meq, mmp;
            eval(mcombos, true, meq, mmp);
            size_t bm = 0;
            for (size_t ci = 1; ci < mcombos.size(); ++ci) {
                if (mmp[ci] > mmp[bm] || (mmp[ci] == mmp[bm] && meq[ci] > meq[bm])) bm = ci;
            }
            last_shooter_.moon_p = static_cast<float>(mmp[bm]);
            last_shooter_.eq_shoot = static_cast<float>(meq[bm]);
            bool commit = true;
            size_t bn = 0;
            std::vector<double> neq, nmp;
            if (cfg_.shooter_mode == "sel") {
                eval(combos, false, neq, nmp);
                for (size_t ci = 1; ci < combos.size(); ++ci) {
                    if (neq[ci] > neq[bn]) bn = ci;
                }
                last_shooter_.eq_norm = static_cast<float>(neq[bn]);
                commit = meq[bm] >= neq[bn];
            }
            const auto& pick = commit ? mcombos[bm] : combos[bn];
            last_shooter_.shooting = commit;
            queued.assign(pick.begin() + 1, pick.end());
            return pick[0];
        }

        std::vector<double> eq, mp;
        eval(combos, false, eq, mp);
        size_t best = 0;
        for (size_t ci = 1; ci < combos.size(); ++ci) {
            if (eq[ci] > eq[best]) best = ci;
        }
        queued.assign(combos[best].begin() + 1, combos[best].end());
        return combos[best][0];
    }

    std::vector<float> PolicyProbs(const HeartsEnv& env, const std::vector<int>& legal) {
        auto obs = env.Observe();
        torch::Tensor o = torch::zeros({1, obs_dim_}, torch::kFloat32);
        FillObsRow(o.data_ptr<float>(), obs, env.GetCurrentPlayer());
        torch::Tensor m = torch::zeros({1, 52}, torch::kBool);
        bool* mp = m.data_ptr<bool>();
        for (int a : legal) mp[a] = true;
        torch::Tensor p = torch::softmax(backend_->Forward(o, m).logits, 1);
        auto acc = p.accessor<float, 2>();
        std::vector<float> probs(legal.size());
        for (size_t i = 0; i < legal.size(); ++i) {
            probs[i] = std::max(acc[0][legal[i]], 1e-6f);
        }
        return probs;
    }

    std::array<int, 3> TopThree(const std::vector<int>& legal, const std::vector<float>& probs) {
        std::vector<size_t> idx(legal.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 3, idx.end(),
                          [&](size_t a, size_t b) { return probs[a] > probs[b]; });
        std::array<int, 3> combo = {legal[idx[0]], legal[idx[1]], legal[idx[2]]};
        std::sort(combo.begin(), combo.end());
        return combo;
    }

    std::array<int, 3> SampleCombo(const std::vector<int>& legal, const std::vector<float>& probs) {
        std::array<int, 3> combo{};
        std::vector<bool> taken(legal.size(), false);
        for (int pick = 0; pick < 3; ++pick) {
            double total = 0.0;
            for (size_t i = 0; i < legal.size(); ++i) {
                if (!taken[i]) total += probs[i];
            }
            std::uniform_real_distribution<double> u(0.0, total);
            double r = u(rng_);
            size_t chosen = 0;
            for (size_t i = 0; i < legal.size(); ++i) {
                if (taken[i]) continue;
                chosen = i;
                if (r < probs[i]) break;
                r -= probs[i];
            }
            taken[chosen] = true;
            combo[pick] = legal[chosen];
        }
        std::sort(combo.begin(), combo.end());
        return combo;
    }

    // ---------------------------- shared internals ----------------------------

    int ArgmaxSingle(const HeartsEnv& env, const std::vector<int>& legal) {
        auto obs = env.Observe();
        torch::Tensor o = torch::zeros({1, obs_dim_}, torch::kFloat32);
        FillObsRow(o.data_ptr<float>(), obs, env.GetCurrentPlayer());
        torch::Tensor m = torch::zeros({1, 52}, torch::kBool);
        bool* mp = m.data_ptr<bool>();
        for (int a : legal) mp[a] = true;
        return backend_->Forward(o, m).logits.argmax(1).item<int>();
    }

    bool FetchBelief(const HeartsEnv& env) {
        auto obs = env.Observe();
        torch::Tensor o = torch::zeros({1, obs_dim_}, torch::kFloat32);
        FillObsRow(o.data_ptr<float>(), obs, env.GetCurrentPlayer());
        torch::Tensor m = torch::ones({1, 52}, torch::kBool);
        InferenceBackend* bk = cfg_.belief_backend ? cfg_.belief_backend.get() : backend_.get();
        InferOutputs out = bk->Forward(o, m);
        if (!out.belief.defined()) return false;
        torch::Tensor probs = torch::sigmoid(out.belief).reshape({3, 52});
        auto acc = probs.accessor<float, 2>();
        for (int k = 0; k < 3; ++k) {
            for (int c = 0; c < 52; ++c) {
                ctx_.belief[k][c] = acc[k][c];
            }
        }
        return true;
    }

    bool TrySample(bool use_belief, std::array<std::vector<int>, 4>& out_hands) {
        std::array<int, 3> caps = ctx_.cap;
        std::vector<int> owner(52, -1);
        std::vector<int> todo;

        for (int c : ctx_.unseen) {
            int pin = ctx_.pinned_rel[c];
            if (pin >= 0) {
                if (caps[pin] <= 0 || ctx_.is_void[pin][c / 13]) return false;
                owner[c] = pin;
                caps[pin]--;
            } else {
                todo.push_back(c);
            }
        }

        // Most-constrained-card-first with weighted owner choice
        while (!todo.empty()) {
            int best_idx = -1, best_count = 4;
            for (size_t i = 0; i < todo.size(); ++i) {
                int c = todo[i], count = 0;
                for (int k = 0; k < 3; ++k) {
                    if (caps[k] > 0 && !ctx_.is_void[k][c / 13]) count++;
                }
                if (count == 0) return false;
                if (count < best_count || (count == best_count && (rng_() & 1))) {
                    best_count = count;
                    best_idx = static_cast<int>(i);
                }
            }
            int c = todo[best_idx];
            todo.erase(todo.begin() + best_idx);

            double w[3], total = 0.0;
            for (int k = 0; k < 3; ++k) {
                bool ok = caps[k] > 0 && !ctx_.is_void[k][c / 13];
                w[k] = ok ? (use_belief ? std::max(ctx_.belief[k][c], 1e-4f) : 1.0) : 0.0;
                total += w[k];
            }
            std::uniform_real_distribution<double> u(0.0, total);
            double r = u(rng_);
            int pick = 2;
            for (int k = 0; k < 3; ++k) {
                if (r < w[k]) { pick = k; break; }
                r -= w[k];
            }
            if (w[pick] == 0.0) return false;
            owner[c] = pick;
            caps[pick]--;
        }

        out_hands[ctx_.me] = ctx_.my_hand;
        for (int c = 0; c < 52; ++c) {
            if (owner[c] >= 0) {
                out_hands[(ctx_.me + owner[c] + 1) % 4].push_back(c);
            }
        }
        return true;
    }

    // Exact fallback sampler. A card's constraints depend only on its suit
    // (voids) and the owners' remaining capacities, so a determinization
    // reduces to a 4-suits x 3-owners transportation problem: how many cards
    // of each suit each opponent takes. DFS over those 12 counts finds a
    // solution whenever one exists; cards within a suit are then dealt out
    // shuffled. No belief weighting - this only runs when the weighted greedy
    // has already failed 200 times.
    bool TrySampleExact(std::array<std::vector<int>, 4>& out_hands) {
        std::array<int, 3> caps = ctx_.cap;
        std::vector<int> owner(52, -1);
        std::array<std::vector<int>, 4> suit_cards;

        for (int c : ctx_.unseen) {
            int pin = ctx_.pinned_rel[c];
            if (pin >= 0) {
                if (caps[pin] <= 0 || ctx_.is_void[pin][c / 13]) return false;
                owner[c] = pin;
                caps[pin]--;
            } else {
                suit_cards[c / 13].push_back(c);
            }
        }

        std::array<int, 4> need{};
        for (int s = 0; s < 4; ++s) need[s] = static_cast<int>(suit_cards[s].size());
        int x[4][3] = {};
        if (!AssignSuits(0, need, caps, x)) return false;

        for (int s = 0; s < 4; ++s) {
            std::shuffle(suit_cards[s].begin(), suit_cards[s].end(), rng_);
            size_t i = 0;
            for (int k = 0; k < 3; ++k) {
                for (int j = 0; j < x[s][k]; ++j) owner[suit_cards[s][i++]] = k;
            }
        }

        out_hands[ctx_.me] = ctx_.my_hand;
        for (int c = 0; c < 52; ++c) {
            if (owner[c] >= 0) {
                out_hands[(ctx_.me + owner[c] + 1) % 4].push_back(c);
            }
        }
        return true;
    }

    bool AssignSuits(int s, const std::array<int, 4>& need, std::array<int, 3>& caps, int x[4][3]) {
        if (s == 4) return true;
        int n = need[s];
        int max0 = ctx_.is_void[0][s] ? 0 : std::min(n, caps[0]);
        for (int i = 0; i <= max0; ++i) {
            int max1 = ctx_.is_void[1][s] ? 0 : std::min(n - i, caps[1]);
            for (int j = 0; j <= max1; ++j) {
                int k = n - i - j;
                if (k > (ctx_.is_void[2][s] ? 0 : caps[2])) continue;
                x[s][0] = i; x[s][1] = j; x[s][2] = k;
                caps[0] -= i; caps[1] -= j; caps[2] -= k;
                if (AssignSuits(s + 1, need, caps, x)) return true;
                caps[0] += i; caps[1] += j; caps[2] += k;
            }
        }
        return false;
    }

    int SetOneHot(int action) {
        last_pi_.fill(0.0f);
        last_pi_[action] = 1.0f;
        return action;
    }

    std::shared_ptr<InferenceBackend> backend_;
    int obs_dim_;
    Config cfg_;
    std::mt19937 rng_;
    Context ctx_;
    std::ofstream probe_out_;
    long probe_decisions_ = 0;
    std::array<double, 4> probe_totals_{};
    int probe_deals_ = 0;
    std::array<std::vector<int>, 4> pending_pass_;  // per-seat queued picks
    std::array<float, 52> last_pi_{};
    ActionStats last_stats_;
    ShooterStats last_shooter_;
};
