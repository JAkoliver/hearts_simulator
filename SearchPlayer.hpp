#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "HeartsEnv.hpp"

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
// At each play decision: sample K complete hidden-hand hypotheses consistent
// with everything observed (hand sizes, observed voids, cards we passed),
// weighted by the network's belief head; then for each legal action, play it
// in every hypothesis and roll all seats out to the end of the round with the
// policy network. Pick the action with the best mean relative score. The same
// K determinizations are shared across actions (common random numbers), so
// the comparison between actions is paired.
class SearchPlayer {
public:
    struct Config {
        int determinizations = 32;
        bool belief_weighted = true;
        unsigned int seed = 12345;
    };

    SearchPlayer(torch::jit::script::Module model, int model_obs_dim, Config cfg = Config())
        : model_(std::move(model)), obs_dim_(model_obs_dim), cfg_(cfg), rng_(cfg.seed) {}

    int ChooseAction(const HeartsEnv& env) {
        std::vector<int> legal = LegalVector(env);
        if (legal.size() == 1) return legal[0];

        // Passing phase: raw policy (pass search deferred)
        if (env.IsPassing()) {
            return ArgmaxSingle(env, legal);
        }

        int me = env.GetCurrentPlayer();

        // Sample K determinizations, shared across candidate actions
        BuildContext(env);
        std::vector<std::array<std::vector<int>, 4>> dets(cfg_.determinizations);
        for (auto& d : dets) d = SampleHands(env);

        struct Sim {
            HeartsEnv sim_env;
            int action_idx;
            bool done;
            Sim(HeartsEnv e, int ai, bool d) : sim_env(std::move(e)), action_idx(ai), done(d) {}
        };
        std::vector<Sim> sims;
        sims.reserve(legal.size() * dets.size());
        for (size_t ai = 0; ai < legal.size(); ++ai) {
            for (const auto& det : dets) {
                HeartsEnv s = env.Clone();
                s.SetHands(det);
                bool done = s.Step(legal[ai]).done;
                sims.emplace_back(std::move(s), static_cast<int>(ai), done);
            }
        }

        // Roll every simulation out to the end of the round, batching all
        // active sims into one forward pass per tick.
        std::vector<size_t> active;
        while (true) {
            active.clear();
            for (size_t i = 0; i < sims.size(); ++i) {
                if (!sims[i].done) active.push_back(i);
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
            torch::Tensor logits;
            {
                torch::NoGradGuard g;
                logits = model_.forward({o, m}).toTuple()->elements()[0].toTensor();
            }
            torch::Tensor acts = logits.argmax(1);
            auto acc = acts.accessor<int64_t, 1>();
            for (size_t j = 0; j < active.size(); ++j) {
                Sim& s = sims[active[j]];
                s.done = s.sim_env.Step(static_cast<int>(acc[j])).done;
            }
        }

        // Aggregate mean relative reward per action
        std::vector<double> score(legal.size(), 0.0);
        for (const auto& s : sims) {
            auto sc = s.sim_env.GetRoundScores();
            double avg = (sc[0] + sc[1] + sc[2] + sc[3]) / 4.0;
            score[s.action_idx] += avg - sc[me];
        }
        size_t best = 0;
        for (size_t ai = 1; ai < legal.size(); ++ai) {
            if (score[ai] > score[best]) best = ai;
        }
        return legal[best];
    }

    // Sample one determinization: full four-hand assignment consistent with
    // all observed constraints. Public so the selftest can validate it.
    // BuildContext(env) must have been called for this state first.
    std::array<std::vector<int>, 4> SampleHands(const HeartsEnv& env) {
        for (int attempt = 0; attempt < 200; ++attempt) {
            bool use_belief = ctx_.have_belief && cfg_.belief_weighted && attempt < 100;
            std::array<std::vector<int>, 4> hands;
            if (TrySample(use_belief, hands)) return hands;
        }
        throw std::runtime_error("SampleHands: could not build a consistent determinization");
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
            int rel = offset;  // receiver = (me + offset) % 4 -> relative index
            for (int a : env.GetPassPicks(ctx_.me)) {
                bool unseen = std::find(ctx_.unseen.begin(), ctx_.unseen.end(), a) != ctx_.unseen.end();
                if (unseen) ctx_.pinned_rel[a] = rel - 1;
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

    std::vector<int> LegalVector(const HeartsEnv& env) {
        std::vector<int> legal;
        auto lr = env.GetLegalActions();
        for (int i = 0; i < 13; ++i) {
            if (lr[i] != -1) legal.push_back(lr[i]);
        }
        return legal;
    }

    int ArgmaxSingle(const HeartsEnv& env, const std::vector<int>& legal) {
        auto obs = env.Observe();
        torch::Tensor o = torch::from_blob((void*)obs.data(), {1, obs_dim_}, torch::kFloat32).clone();
        torch::Tensor m = torch::zeros({1, 52}, torch::kBool);
        bool* mp = m.data_ptr<bool>();
        for (int a : legal) mp[a] = true;
        torch::NoGradGuard g;
        auto logits = model_.forward({o, m}).toTuple()->elements()[0].toTensor();
        return logits.argmax(1).item<int>();
    }

    bool FetchBelief(const HeartsEnv& env) {
        auto obs = env.Observe();
        torch::Tensor o = torch::from_blob((void*)obs.data(), {1, obs_dim_}, torch::kFloat32).clone();
        torch::Tensor m = torch::ones({1, 52}, torch::kBool);
        torch::NoGradGuard g;
        auto out = model_.forward({o, m}).toTuple();
        if (out->elements().size() < 3) return false;
        torch::Tensor probs = torch::sigmoid(out->elements()[2].toTensor()).reshape({3, 52});
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
                if (caps[pin] <= 0 || ctx_.is_void[pin][c / 13]) return false;  // impossible pin: bug upstream
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
                if (count == 0) return false;  // dead end: retry with a fresh pass
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

    torch::jit::script::Module model_;
    int obs_dim_;
    Config cfg_;
    std::mt19937 rng_;
    Context ctx_;
};
