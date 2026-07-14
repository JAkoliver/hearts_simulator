#pragma once

// Batched inference for self-play generation.
//
// Many small forward passes (K x legal rows) pay a fixed kernel-launch +
// sync cost that dwarfs the actual compute for a ~2M-param net, and launches
// from separate processes serialize at the GPU. The fix: deal-playing threads
// submit requests to ONE server thread that concatenates everything waiting
// into a single forward. No batch-window timers - while a forward is in
// flight new requests pile up, so batch size adapts to load automatically.
//
// InferenceBackend  - interface SearchPlayer talks to
// DirectBackend     - immediate forward on a given device (single-threaded use)
// InferenceServer   - owns the module + service thread
// ServedBackend     - client handle submitting to an InferenceServer

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <torch/script.h>

struct InferOutputs {
    // CPU tensors; belief is undefined for 2-output (policy+value) traces.
    torch::Tensor logits;
    torch::Tensor value;
    torch::Tensor belief;
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    virtual InferOutputs Forward(const torch::Tensor& obs, const torch::Tensor& mask) = 0;
};

namespace infer_detail {
inline InferOutputs Unpack(const c10::intrusive_ptr<c10::ivalue::Tuple>& out) {
    InferOutputs res;
    res.logits = out->elements()[0].toTensor().to(torch::kCPU);
    res.value = out->elements()[1].toTensor().to(torch::kCPU);
    if (out->elements().size() >= 3) {
        res.belief = out->elements()[2].toTensor().to(torch::kCPU);
    }
    return res;
}
}  // namespace infer_detail

class DirectBackend : public InferenceBackend {
public:
    DirectBackend(torch::jit::script::Module module, torch::Device device)
        : module_(std::move(module)), device_(device) {
        module_.to(device_);
        module_.eval();
    }

    InferOutputs Forward(const torch::Tensor& obs, const torch::Tensor& mask) override {
        torch::NoGradGuard g;
        auto out = module_.forward({obs.to(device_), mask.to(device_)}).toTuple();
        return infer_detail::Unpack(out);
    }

private:
    torch::jit::script::Module module_;
    torch::Device device_;
};

class InferenceServer {
public:
    InferenceServer(torch::jit::script::Module module, torch::Device device)
        : module_(std::move(module)), device_(device) {
        module_.to(device_);
        module_.eval();
        worker_ = std::thread([this] { Loop(); });
    }

    ~InferenceServer() {
        {
            std::lock_guard<std::mutex> g(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        worker_.join();
    }

    InferOutputs Submit(const torch::Tensor& obs, const torch::Tensor& mask) {
        Request r;
        r.obs = obs;
        r.mask = mask;
        std::future<InferOutputs> fut = r.promise.get_future();
        {
            std::lock_guard<std::mutex> g(mu_);
            queue_.push_back(std::move(r));
        }
        cv_.notify_one();
        return fut.get();  // rethrows if the forward failed
    }

    long Launches() const { return launches_.load(); }
    double MeanBatchRows() const {
        long l = launches_.load();
        return l ? static_cast<double>(rows_.load()) / l : 0.0;
    }

private:
    struct Request {
        torch::Tensor obs, mask;
        std::promise<InferOutputs> promise;
    };

    void Loop() {
        torch::NoGradGuard g;
        std::vector<Request> batch;
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) return;  // stop requested and drained
                batch.clear();
                batch.swap(queue_);
            }
            torch::Tensor o, m;
            if (batch.size() == 1) {
                o = batch[0].obs;
                m = batch[0].mask;
            } else {
                std::vector<torch::Tensor> os, ms;
                os.reserve(batch.size());
                ms.reserve(batch.size());
                for (const auto& r : batch) {
                    os.push_back(r.obs);
                    ms.push_back(r.mask);
                }
                o = torch::cat(os, 0);
                m = torch::cat(ms, 0);
            }
            try {
                auto out = module_.forward({o.to(device_), m.to(device_)}).toTuple();
                InferOutputs all = infer_detail::Unpack(out);
                launches_.fetch_add(1);
                rows_.fetch_add(all.logits.size(0));
                int64_t row = 0;
                for (auto& r : batch) {
                    int64_t n = r.obs.size(0);
                    InferOutputs res;
                    res.logits = all.logits.slice(0, row, row + n);
                    res.value = all.value.slice(0, row, row + n);
                    if (all.belief.defined()) {
                        res.belief = all.belief.slice(0, row, row + n);
                    }
                    r.promise.set_value(std::move(res));
                    row += n;
                }
            } catch (...) {
                for (auto& r : batch) {
                    r.promise.set_exception(std::current_exception());
                }
            }
        }
    }

    torch::jit::script::Module module_;
    torch::Device device_;
    std::vector<Request> queue_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::thread worker_;
    std::atomic<long> launches_{0};
    std::atomic<long> rows_{0};
};

class ServedBackend : public InferenceBackend {
public:
    explicit ServedBackend(InferenceServer* server) : server_(server) {}

    InferOutputs Forward(const torch::Tensor& obs, const torch::Tensor& mask) override {
        return server_->Submit(obs, mask);
    }

private:
    InferenceServer* server_;
};
