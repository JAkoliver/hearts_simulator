#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Magically translates std::array to Python lists
#include "HeartsEnv.hpp"

namespace py = pybind11;

// "hearts_env" will be the name of the module you import in Python
PYBIND11_MODULE(hearts_env, m) {
    m.doc() = "High-Performance C++ Hearts Environment for RL";

    // 1. Expose the StepResult struct so Python can read the reward and done flags
    py::class_<StepResult>(m, "StepResult")
        .def_readonly("reward", &StepResult::reward)
        .def_readonly("done", &StepResult::done);

    // 2. Expose the HeartsEnv class and its core methods
    py::class_<HeartsEnv>(m, "HeartsEnv")
        .def(py::init<unsigned int>(), py::arg("seed") = 42)
        .def("reset", &HeartsEnv::Reset)
        .def("step", &HeartsEnv::Step, py::arg("action_id"))
        .def("get_legal_actions", &HeartsEnv::GetLegalActions)
        .def("observe", &HeartsEnv::Observe)
        .def("get_round_scores", &HeartsEnv::GetRoundScores)
        .def("get_current_player", &HeartsEnv::GetCurrentPlayer);
}
