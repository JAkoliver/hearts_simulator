#pragma once

// TreeSearchPlayer: single-observer ISMCTS with PUCT priors and Max^n backup.
//
// The flat SearchPlayer is one ply deep: it scores each legal action by the
// mean of K terminal rollouts and picks the max. Its measured ceiling (the
// K-curve saturates at 64) is the ceiling of that structure, not of search
// itself. This player builds a TREE over the information set:
//
//   - Every iteration samples a fresh belief-weighted determinization
//     (reusing SearchPlayer's sampler) and descends one shared tree.
//   - At each node the ACTING seat picks argmax of
//         Q_seat(child) + c_puct * P(child) * sqrt(N_parent) / (1 + N_child)
//     restricted to actions legal in the current determinization
//     (subset-armed ISMCTS; priors renormalized over the legal subset).
//     Hearts is 4-player and non-zero-sum, so nodes carry a per-seat value
//     vector and each seat maximizes its own component (Max^n backup).
//   - Unexpanded leaves are expanded with policy-net priors and evaluated by
//     batched argmax rollouts to the end of the round - the proven unbiased
//     evaluator (learned leaf evaluators are measured unusable here).
//   - Leaf parallelism with virtual loss: leaf_batch descents run per wave,
//     then ONE batched prior forward + ONE batched rollout serve them all.
//
// Values are normalized to points/13 so c_puct lives on a familiar scale.
// The move played is the most-visited root child; LastPolicy() exposes the
// root visit distribution (the AlphaZero-style distillation target).
//
// Passing phase delegates to the flat player's pass search unchanged.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <torch/script.h>

#include "HeartsEnv.hpp"
#include "InferenceServer.hpp"
#include "SearchPlayer.hpp"

class TreeSearchPlayer : public IPlayer {
public:
    struct Config {
        int iterations = 400;
        float c_puct = 1.5f;
        int leaf_batch = 16;
        // Discourages same-path collisions inside a wave (normalized units)
        float virtual_loss = 0.3f;
        unsigned int seed = 12345;
        // Root Dirichlet noise (off for evaluation; on for self-play diversity)
        bool root_noise = false;
        float dirichlet_alpha = 0.3f;
        float noise_frac = 0.25f;
        // Settings for the delegated passing-phase flat search (and sampler)
        SearchPlayer::Config pass_cfg;
    };

    TreeSearchPlayer(std::shared_ptr<InferenceBackend> backend, int model_obs_dim, Config cfg)
        : backend_(std::move(backend)), obs_dim_(model_obs_dim), cfg_(cfg), rng_(cfg.seed),
          flat_(backend_, model_obs_dim, cfg.pass_cfg) {}

    TreeSearchPlayer(torch::jit::script::Module model, int model_obs_dim, Config cfg)
        : TreeSearchPlayer(std::make_shared<DirectBackend>(std::move(model), cfg.pass_cfg.device,
                                                           cfg.pass_cfg.bf16),
                           model_obs_dim, cfg) {}

    int ChooseAction(const HeartsEnv& env) override {
        std::vector<int> legal = LegalVec(env);
        if (env.IsPassing()) {
            int a = flat_.ChooseAction(env);
            last_pi_ = flat_.LastPolicy();
            return a;
        }
        if (legal.size() == 1) return SetOneHot(legal[0]);

        const int me = env.GetCurrentPlayer();
        nodes_.clear();
        nodes_.emplace_back();
        nodes_[0].acting_seat = me;

        flat_.BuildContext(env);  // sampler context (belief, voids, pins)

        // Root expansion uses the REAL observation
        {
            torch::Tensor o = ObsTensor(env);
            torch::Tensor m = MaskTensor(legal);
            InferOutputs out = backend_->Forward(o, m);
            SetPriors(0, out.logits);
            nodes_[0].expanded = true;
            if (cfg_.root_noise) ApplyDirichletNoise(legal);
        }

        struct Leaf {
            int node;
            HeartsEnv sim;
            Leaf(int n, HeartsEnv s) : node(n), sim(std::move(s)) {}
        };

        int completed = 0;
        std::vector<Leaf> leaves;
        while (completed < cfg_.iterations) {
            leaves.clear();
            int wave = std::min(cfg_.leaf_batch, cfg_.iterations - completed);
            for (int d = 0; d < wave; ++d) {
                HeartsEnv sim = env.Clone();
                sim.SetHands(flat_.SampleHands(env));
                int cur = 0;
                bool terminal = false;
                std::array<double, 4> tv{};
                while (nodes_[cur].expanded) {
                    int a = SelectChild(cur, sim);
                    int ci = nodes_[cur].child[a];
                    if (ci < 0) ci = NewChild(cur, a);  // may reallocate nodes_
                    nodes_[ci].vloss++;
                    bool done = sim.Step(a).done;
                    cur = ci;
                    if (done) {
                        terminal = true;
                        for (int s = 0; s < 4; ++s) tv[s] = NormReward(sim, s);
                        break;
                    }
                }
                if (terminal) {
                    Backup(cur, tv);
                    completed++;
                } else {
                    leaves.emplace_back(cur, std::move(sim));
                }
            }
            if (leaves.empty()) continue;

            // Batch-expand all unexpanded leaf nodes with policy priors
            ExpandLeaves(leaves);
            // Batch-rollout every leaf sim to the end of the round
            RolloutLeaves(leaves);
            for (auto& lf : leaves) {
                std::array<double, 4> v{};
                for (int s = 0; s < 4; ++s) v[s] = NormReward(lf.sim, s);
                Backup(lf.node, v);
            }
            completed += static_cast<int>(leaves.size());
        }

        // Most-visited root move; visit distribution is the teacher target
        last_pi_.fill(0.0f);
        int best = legal[0], best_n = -1;
        double total = 0.0;
        for (int a : legal) {
            int ci = nodes_[0].child[a];
            int n = (ci >= 0) ? nodes_[ci].n : 0;
            total += n;
            if (n > best_n) {
                best_n = n;
                best = a;
            }
        }
        if (total > 0) {
            for (int a : legal) {
                int ci = nodes_[0].child[a];
                last_pi_[a] = (ci >= 0) ? static_cast<float>(nodes_[ci].n / total) : 0.0f;
            }
        } else {
            last_pi_[best] = 1.0f;
        }
        return best;
    }

    const std::array<float, 52>& LastPolicy() const override { return last_pi_; }

    // Total visits at the root of the last search (selftest support)
    int LastRootVisits() const { return nodes_.empty() ? 0 : nodes_[0].n; }

private:
    struct Node {
        int parent = -1;
        bool expanded = false;
        int acting_seat = -1;
        int n = 0;
        int vloss = 0;
        std::array<double, 4> w{};       // per-seat value sums, points/13
        std::array<float, 52> priors{};  // cached at expansion
        std::array<int, 52> child;       // action -> node index, -1 = none

        Node() { child.fill(-1); }
    };

    static std::vector<int> LegalVec(const HeartsEnv& env) {
        std::vector<int> legal;
        auto lr = env.GetLegalActions();
        for (int i = 0; i < 13; ++i) {
            if (lr[i] != -1) legal.push_back(lr[i]);
        }
        return legal;
    }

    static double NormReward(const HeartsEnv& env, int seat) {
        auto sc = env.GetRoundScores();
        double avg = (sc[0] + sc[1] + sc[2] + sc[3]) / 4.0;
        return (avg - sc[seat]) / 13.0;
    }

    torch::Tensor ObsTensor(const HeartsEnv& env) {
        auto obs = env.Observe();
        torch::Tensor o = torch::empty({1, obs_dim_}, torch::kFloat32);
        std::memcpy(o.data_ptr<float>(), obs.data(), obs_dim_ * sizeof(float));
        return o;
    }

    static torch::Tensor MaskTensor(const std::vector<int>& legal) {
        torch::Tensor m = torch::zeros({1, 52}, torch::kBool);
        bool* mp = m.data_ptr<bool>();
        for (int a : legal) mp[a] = true;
        return m;
    }

    void SetPriors(int ni, const torch::Tensor& logits_row) {
        torch::Tensor p = torch::softmax(logits_row, 1);
        auto acc = p.accessor<float, 2>();
        for (int a = 0; a < 52; ++a) nodes_[ni].priors[a] = acc[0][a];
    }

    void ApplyDirichletNoise(const std::vector<int>& legal) {
        std::gamma_distribution<double> gamma(cfg_.dirichlet_alpha, 1.0);
        std::vector<double> noise(legal.size());
        double sum = 0.0;
        for (auto& x : noise) {
            x = gamma(rng_);
            sum += x;
        }
        if (sum <= 0.0) return;
        for (size_t i = 0; i < legal.size(); ++i) {
            float& pr = nodes_[0].priors[legal[i]];
            pr = (1.0f - cfg_.noise_frac) * pr
                 + cfg_.noise_frac * static_cast<float>(noise[i] / sum);
        }
    }

    int SelectChild(int ni, const HeartsEnv& sim) {
        const Node& node = nodes_[ni];
        int seat = sim.GetCurrentPlayer();
        auto lr = sim.GetLegalActions();

        double psum = 0.0;
        for (int i = 0; i < 13; ++i) {
            if (lr[i] != -1) psum += std::max(node.priors[lr[i]], 1e-4f);
        }
        double sqrtN = std::sqrt(static_cast<double>(node.n + node.vloss) + 1.0);

        int best = -1;
        double best_score = -1e18;
        for (int i = 0; i < 13; ++i) {
            int a = lr[i];
            if (a == -1) continue;
            double q = 0.0, nc = 0.0;
            int ci = node.child[a];
            if (ci >= 0) {
                const Node& c = nodes_[ci];
                nc = c.n + c.vloss;
                if (nc > 0) {
                    q = (c.w[seat] - cfg_.virtual_loss * c.vloss) / nc;
                }
            }
            double p = std::max(node.priors[a], 1e-4f) / psum;
            double score = q + cfg_.c_puct * p * sqrtN / (1.0 + nc);
            if (score > best_score) {
                best_score = score;
                best = a;
            }
        }
        return best;
    }

    int NewChild(int parent, int action) {
        int idx = static_cast<int>(nodes_.size());
        nodes_.emplace_back();
        nodes_[idx].parent = parent;
        nodes_[parent].child[action] = idx;
        return idx;
    }

    void Backup(int leaf, const std::array<double, 4>& v) {
        for (int i = leaf; i != -1; i = nodes_[i].parent) {
            Node& node = nodes_[i];
            node.n++;
            for (int s = 0; s < 4; ++s) node.w[s] += v[s];
            if (i != 0 && node.vloss > 0) node.vloss--;
        }
    }

    template <typename LeafVec>
    void ExpandLeaves(LeafVec& leaves) {
        // Unique unexpanded nodes only (two descents can reach the same node)
        std::vector<size_t> need;
        for (size_t j = 0; j < leaves.size(); ++j) {
            int ni = leaves[j].node;
            if (!nodes_[ni].expanded) {
                nodes_[ni].expanded = true;  // claims it; priors set below
                nodes_[ni].acting_seat = leaves[j].sim.GetCurrentPlayer();
                need.push_back(j);
            }
        }
        if (need.empty()) return;

        torch::Tensor o = torch::empty({(long)need.size(), obs_dim_}, torch::kFloat32);
        torch::Tensor m = torch::zeros({(long)need.size(), 52}, torch::kBool);
        float* op = o.data_ptr<float>();
        bool* mp = m.data_ptr<bool>();
        for (size_t k = 0; k < need.size(); ++k) {
            const HeartsEnv& sim = leaves[need[k]].sim;
            auto obs = sim.Observe();
            std::memcpy(op + k * obs_dim_, obs.data(), obs_dim_ * sizeof(float));
            auto lr = sim.GetLegalActions();
            for (int i = 0; i < 13; ++i) {
                if (lr[i] != -1) mp[k * 52 + lr[i]] = true;
            }
        }
        torch::Tensor probs = torch::softmax(backend_->Forward(o, m).logits, 1);
        auto acc = probs.accessor<float, 2>();
        for (size_t k = 0; k < need.size(); ++k) {
            Node& node = nodes_[leaves[need[k]].node];
            for (int a = 0; a < 52; ++a) node.priors[a] = acc[k][a];
        }
    }

    template <typename LeafVec>
    void RolloutLeaves(LeafVec& leaves) {
        std::vector<size_t> active;
        std::vector<bool> done(leaves.size(), false);
        for (size_t j = 0; j < leaves.size(); ++j) {
            // A leaf sim cannot be terminal here (terminals backed up inline)
            done[j] = false;
        }
        while (true) {
            active.clear();
            for (size_t j = 0; j < leaves.size(); ++j) {
                if (!done[j]) active.push_back(j);
            }
            if (active.empty()) return;

            torch::Tensor o = torch::empty({(long)active.size(), obs_dim_}, torch::kFloat32);
            torch::Tensor m = torch::zeros({(long)active.size(), 52}, torch::kBool);
            float* op = o.data_ptr<float>();
            bool* mp = m.data_ptr<bool>();
            for (size_t k = 0; k < active.size(); ++k) {
                const HeartsEnv& sim = leaves[active[k]].sim;
                auto obs = sim.Observe();
                std::memcpy(op + k * obs_dim_, obs.data(), obs_dim_ * sizeof(float));
                auto lr = sim.GetLegalActions();
                for (int i = 0; i < 13; ++i) {
                    if (lr[i] != -1) mp[k * 52 + lr[i]] = true;
                }
            }
            torch::Tensor acts = backend_->Forward(o, m).logits.argmax(1);
            auto acc = acts.accessor<int64_t, 1>();
            for (size_t k = 0; k < active.size(); ++k) {
                done[active[k]] = leaves[active[k]].sim.Step(static_cast<int>(acc[k])).done;
            }
        }
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
    SearchPlayer flat_;  // sampler, pass search, and backend sharing
    std::vector<Node> nodes_;
    std::array<float, 52> last_pi_{};
};
