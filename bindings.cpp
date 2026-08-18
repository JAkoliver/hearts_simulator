#include <cstring>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Magically translates std::array to Python lists
#include "HeartsEnv.hpp"

namespace py = pybind11;

// Batched environment pool. The single-env API marshals every observation
// into a Python list of 550 floats (~100us of pure conversion per decision,
// ~15M decisions per training trial - the measured rollout bottleneck).
// HeartsVecEnv owns N envs and serves batched numpy arrays written directly
// from C++; Python keeps the torch net, seat assignment, and buffers.
class HeartsVecEnv {
public:
    HeartsVecEnv(int n, unsigned int seed0, bool enable_passing = true) {
        envs_.reserve(n);
        pending_block_.assign(n, {-1, 0});   // Addendum R
        for (int i = 0; i < n; ++i) {
            envs_.emplace_back(seed0 + i, enable_passing);
            envs_.back().Reset();  // a freshly constructed env has no deal yet
        }
    }

    int size() const { return static_cast<int>(envs_.size()); }

    void reset_all() {
        for (auto& e : envs_) e.Reset();
    }

    py::array_t<int32_t> current_players() const {
        py::array_t<int32_t> out(static_cast<py::ssize_t>(envs_.size()));
        auto o = out.mutable_unchecked<1>();
        for (size_t i = 0; i < envs_.size(); ++i) {
            o(i) = envs_[i].GetCurrentPlayer();
        }
        return out;
    }

    py::array_t<float> observe_batch(py::array_t<int64_t> idx) const {
        auto ix = idx.unchecked<1>();
        py::array_t<float> out({ix.shape(0), (py::ssize_t)550});
        auto o = out.mutable_unchecked<2>();
        for (py::ssize_t j = 0; j < ix.shape(0); ++j) {
            auto obs = envs_[Check(ix(j))].Observe();
            std::memcpy(o.mutable_data(j, 0), obs.data(), 550 * sizeof(float));
        }
        return out;
    }

    // obs-v2 extension block (326 dims past the classic 550+match6), for
    // v6 learners in the vec PPO loop (docs/v6_prereg.md stage 4). Same
    // per-index contract as observe_batch.
    py::array_t<float> observe_ext_batch(py::array_t<int64_t> idx) const {
        auto ix = idx.unchecked<1>();
        py::array_t<float> out({ix.shape(0), (py::ssize_t)326});
        auto o = out.mutable_unchecked<2>();
        for (py::ssize_t j = 0; j < ix.shape(0); ++j) {
            auto ext = envs_[Check(ix(j))].ObserveExt();
            std::memcpy(o.mutable_data(j, 0), ext.data(),
                        326 * sizeof(float));
        }
        return out;
    }

    py::array_t<bool> legal_mask_batch(py::array_t<int64_t> idx) const {
        auto ix = idx.unchecked<1>();
        py::array_t<bool> out({ix.shape(0), (py::ssize_t)52});
        auto o = out.mutable_unchecked<2>();
        for (py::ssize_t j = 0; j < ix.shape(0); ++j) {
            for (int a = 0; a < 52; ++a) o(j, a) = false;
            auto lr = envs_[Check(ix(j))].GetLegalActions();
            for (int i = 0; i < 13; ++i) {
                if (lr[i] != -1) o(j, lr[i]) = true;
            }
        }
        return out;
    }

    py::array_t<float> labels_batch(py::array_t<int64_t> idx) const {
        auto ix = idx.unchecked<1>();
        py::array_t<float> out({ix.shape(0), (py::ssize_t)156});
        auto o = out.mutable_unchecked<2>();
        for (py::ssize_t j = 0; j < ix.shape(0); ++j) {
            auto lab = envs_[Check(ix(j))].ObserveOpponentHands();
            std::memcpy(o.mutable_data(j, 0), lab.data(), 156 * sizeof(float));
        }
        return out;
    }

    // Steps each env; done envs report their final round scores and are
    // auto-reset AFTER the scores are captured (the caller patches terminal
    // rewards from the returned scores, then reassigns seats).
    py::tuple step_batch(py::array_t<int64_t> idx, py::array_t<int64_t> actions) {
        auto ix = idx.unchecked<1>();
        auto ac = actions.unchecked<1>();
        if (ix.shape(0) != ac.shape(0)) {
            throw std::runtime_error("step_batch: idx and actions length mismatch");
        }
        py::array_t<bool> dones(ix.shape(0));
        py::array_t<int32_t> scores({ix.shape(0), (py::ssize_t)4});
        auto d = dones.mutable_unchecked<1>();
        auto s = scores.mutable_unchecked<2>();
        for (py::ssize_t j = 0; j < ix.shape(0); ++j) {
            HeartsEnv& env = envs_[Check(ix(j))];
            bool done = env.Step(static_cast<int>(ac(j))).done;
            d(j) = done;
            // Addendum R: capture the block event (if any) BEFORE the
            // deal-end auto-reset below can clear it; pending until read.
            {
                auto ev = env.TakeBlockEvent();
                if (ev.first >= 0) pending_block_[Check(ix(j))] = ev;
            }
            if (done) {
                auto sc = env.GetRoundScores();
                for (int i = 0; i < 4; ++i) s(j, i) = sc[i];
                env.Reset();
            } else {
                for (int i = 0; i < 4; ++i) s(j, i) = 0;
            }
        }
        return py::make_tuple(dones, scores);
    }

private:
    size_t Check(int64_t i) const {
        if (i < 0 || i >= static_cast<int64_t>(envs_.size())) {
            throw std::out_of_range("HeartsVecEnv: env index out of range");
        }
        return static_cast<size_t>(i);
    }

    std::vector<HeartsEnv> envs_;
    // Addendum R: per-env pending block event {seat, pts}, seat -1 = none
    std::vector<std::pair<int, int>> pending_block_;

public:
    // Addendum R: (n,2) int32 [blocker seat, penalty pts] for the requested
    // envs since their last read (seat -1 = none); read-and-clear.
    py::array_t<int32_t> block_events_batch(py::array_t<int64_t> idx) {
        auto ix = idx.unchecked<1>();
        py::array_t<int32_t> out({ix.shape(0), (py::ssize_t)2});
        auto o = out.mutable_unchecked<2>();
        for (py::ssize_t j = 0; j < ix.shape(0); ++j) {
            auto& ev = pending_block_[Check(ix(j))];
            o(j, 0) = ev.first; o(j, 1) = ev.second;
            ev = {-1, 0};
        }
        return out;
    }
};

// "hearts_env" will be the name of the module you import in Python
PYBIND11_MODULE(hearts_env, m) {
    m.doc() = "High-Performance C++ Hearts Environment for RL";

    // 1. Expose the StepResult struct so Python can read the reward and done flags
    py::class_<StepResult>(m, "StepResult")
        .def_readonly("reward", &StepResult::reward)
        .def_readonly("done", &StepResult::done);

    // 2. Expose the HeartsEnv class and its core methods
    // Passing defaults ON for the Python training/eval side; the C++ game
    // (main.cpp) constructs with seed only and keeps passing disabled.
    py::class_<HeartsEnv>(m, "HeartsEnv")
        .def(py::init<unsigned int, bool>(), py::arg("seed") = 42, py::arg("enable_passing") = true)
        .def("reset", &HeartsEnv::Reset)
        .def("step", &HeartsEnv::Step, py::arg("action_id"))
        .def("get_legal_actions", &HeartsEnv::GetLegalActions)
        .def("observe", &HeartsEnv::Observe)
        .def("observe_for", &HeartsEnv::ObserveFor)
        .def("observe_ext", &HeartsEnv::ObserveExt)
        .def("observe_ext_for", &HeartsEnv::ObserveExtFor)
        .def("get_round_scores", &HeartsEnv::GetRoundScores)
        .def("get_current_player", &HeartsEnv::GetCurrentPlayer)
        .def("is_passing", &HeartsEnv::IsPassing)
        .def("get_pass_direction", &HeartsEnv::GetPassDirection)
        .def("observe_opponent_hands", &HeartsEnv::ObserveOpponentHands)
        // Cross-toolchain replay: seed-based dealing is stdlib-shuffle
        // implementation-bound (see HeartsEnv.hpp SetDeal comment), so
        // replays of logs produced by a different build install the
        // logged hands instead of trusting the seed.
        .def("set_deal", &HeartsEnv::SetDeal, py::arg("hand_ids"))
        .def("take_block_event", &HeartsEnv::TakeBlockEvent);

    // 3. Batched environment pool (numpy in/out; see class comment)
    py::class_<HeartsVecEnv>(m, "HeartsVecEnv")
        .def(py::init<int, unsigned int, bool>(),
             py::arg("n"), py::arg("seed0") = 1000, py::arg("enable_passing") = true)
        .def("size", &HeartsVecEnv::size)
        .def("reset_all", &HeartsVecEnv::reset_all)
        .def("current_players", &HeartsVecEnv::current_players)
        .def("observe_batch", &HeartsVecEnv::observe_batch)
        .def("observe_ext_batch", &HeartsVecEnv::observe_ext_batch)
        .def("legal_mask_batch", &HeartsVecEnv::legal_mask_batch)
        .def("labels_batch", &HeartsVecEnv::labels_batch)
        .def("step_batch", &HeartsVecEnv::step_batch)
        .def("block_events_batch", &HeartsVecEnv::block_events_batch);
}
