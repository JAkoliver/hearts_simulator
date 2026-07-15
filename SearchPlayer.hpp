#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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
    for (int dim : {550, 238, 181}) {
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
class SearchPlayer {
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
        // Optional separate net for belief marginals (determinization
        // sampling only): lets a strong rollout policy borrow a
        // better-calibrated belief head from another checkpoint. Null =
        // use the main backend.
        std::shared_ptr<InferenceBackend> belief_backend;
    };

    // Shared-backend constructor: many players (threads) funnel inference
    // through one backend, e.g. a ServedBackend on an InferenceServer.
    SearchPlayer(std::shared_ptr<InferenceBackend> backend, int model_obs_dim,
                 Config cfg = Config())
        : backend_(std::move(backend)), obs_dim_(model_obs_dim), cfg_(cfg), rng_(cfg.seed) {}

    SearchPlayer(torch::jit::script::Module model, int model_obs_dim, Config cfg = Config())
        : SearchPlayer(std::make_shared<DirectBackend>(std::move(model), cfg.device),
                       model_obs_dim, cfg) {}

    int ChooseAction(const HeartsEnv& env) {
        std::vector<int> legal = LegalVector(env);
        if (env.IsPassing()) return SetOneHot(ChoosePass(env, legal));
        if (legal.size() == 1) return SetOneHot(legal[0]);

        int me = env.GetCurrentPlayer();

        // Sample K determinizations, shared across candidate actions
        BuildContext(env);
        std::vector<std::array<std::vector<int>, 4>> dets(cfg_.determinizations);
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

        std::vector<double> score(legal.size(), 0.0);
        for (const auto& s : sims) {
            score[s.tag] += s.result;
        }
        size_t best = 0;
        for (size_t ai = 1; ai < legal.size(); ++ai) {
            if (score[ai] > score[best]) best = ai;
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
    const std::array<float, 52>& LastPolicy() const { return last_pi_; }

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

        Sim(HeartsEnv e, int t) : sim_env(std::move(e)), tag(t) {}
    };

    static double RelReward(const HeartsEnv& env, int seat) {
        auto sc = env.GetRoundScores();
        double avg = (sc[0] + sc[1] + sc[2] + sc[3]) / 4.0;
        return avg - sc[seat];
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
            // Consume any scripted picks first (no inference needed)
            bool progressed = true;
            while (progressed) {
                progressed = false;
                for (auto& s : sims) {
                    if (!s.done && s.script_pos < s.script.size() && s.sim_env.IsPassing()
                        && s.sim_env.GetCurrentPlayer() == s.script_seat) {
                        s.done = s.sim_env.Step(s.script[s.script_pos++]).done;
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
                std::memcpy(op + j * obs_dim_, obs.data(), obs_dim_ * sizeof(float));
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
        // batched through the value head in one forward.
        std::vector<size_t> leaves;
        for (size_t i = 0; i < sims.size(); ++i) {
            if (sims[i].done) {
                sims[i].result = RelReward(sims[i].sim_env, sims[i].eval_seat);
            } else {
                leaves.push_back(i);
            }
        }
        if (leaves.empty()) return;

        torch::Tensor o = torch::empty({(long)leaves.size(), obs_dim_}, torch::kFloat32);
        float* op = o.data_ptr<float>();
        for (size_t j = 0; j < leaves.size(); ++j) {
            Sim& s = sims[leaves[j]];
            auto obs = s.sim_env.ObserveFor(s.eval_seat);
            std::memcpy(op + j * obs_dim_, obs.data(), obs_dim_ * sizeof(float));
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

        std::vector<Sim> sims;
        sims.reserve(combos.size() * dets.size());
        for (size_t ci = 0; ci < combos.size(); ++ci) {
            for (const auto& det : dets) {
                Sim s(env.Clone(), static_cast<int>(ci));
                s.sim_env.ResetForPassSearch(det);
                s.eval_seat = me;
                s.start_tricks = 0;  // pass search rewinds to the deal start
                s.script.assign(combos[ci].begin(), combos[ci].end());
                s.script_seat = me;
                sims.push_back(std::move(s));
            }
        }
        RolloutAll(sims);

        std::vector<double> score(combos.size(), 0.0);
        for (const auto& s : sims) {
            score[s.tag] += s.result;
        }
        size_t best = 0;
        for (size_t ci = 1; ci < combos.size(); ++ci) {
            if (score[ci] > score[best]) best = ci;
        }
        queued.assign(combos[best].begin() + 1, combos[best].end());
        return combos[best][0];
    }

    std::vector<float> PolicyProbs(const HeartsEnv& env, const std::vector<int>& legal) {
        auto obs = env.Observe();
        torch::Tensor o = torch::from_blob((void*)obs.data(), {1, obs_dim_}, torch::kFloat32).clone();
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
        torch::Tensor o = torch::from_blob((void*)obs.data(), {1, obs_dim_}, torch::kFloat32).clone();
        torch::Tensor m = torch::zeros({1, 52}, torch::kBool);
        bool* mp = m.data_ptr<bool>();
        for (int a : legal) mp[a] = true;
        return backend_->Forward(o, m).logits.argmax(1).item<int>();
    }

    bool FetchBelief(const HeartsEnv& env) {
        auto obs = env.Observe();
        torch::Tensor o = torch::from_blob((void*)obs.data(), {1, obs_dim_}, torch::kFloat32).clone();
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
    std::array<std::vector<int>, 4> pending_pass_;  // per-seat queued picks
    std::array<float, 52> last_pi_{};
};
