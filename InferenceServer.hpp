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

#include <ATen/autocast_mode.h>
#include <torch/script.h>

// RAII guard: bf16 autocast for CUDA inference. The module stays fp32 (a
// traced module converted wholesale to bf16 crashes natively); autocast
// runs the matmul-heavy ops in bf16 the same way train.py does.
class AutocastGuard {
public:
    explicit AutocastGuard(bool enable) : enabled_(enable) {
        if (enabled_) {
            at::autocast::set_autocast_enabled(at::kCUDA, true);
            at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
        }
    }
    ~AutocastGuard() {
        if (enabled_) {
            at::autocast::clear_cache();
            at::autocast::set_autocast_enabled(at::kCUDA, false);
        }
    }

private:
    bool enabled_;
};

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
    // Oracle value head: expected outcome GIVEN hands (leaf evaluation for
    // determinized search). Only available if the traced module exposes an
    // "oracle" method.
    virtual bool HasOracle() const { return false; }
    virtual torch::Tensor OracleForward(const torch::Tensor& /*obs*/,
                                        const torch::Tensor& /*hands*/) {
        throw std::runtime_error("backend has no oracle method");
    }
};

namespace infer_detail {
inline torch::Tensor ToHost(const torch::Tensor& t) {
    // Upcast covers bf16 inference; a no-op for fp32
    return t.to(torch::kFloat).to(torch::kCPU);
}

inline InferOutputs Unpack(const c10::intrusive_ptr<c10::ivalue::Tuple>& out) {
    InferOutputs res;
    res.logits = ToHost(out->elements()[0].toTensor());
    res.value = ToHost(out->elements()[1].toTensor());
    if (out->elements().size() >= 3) {
        res.belief = ToHost(out->elements()[2].toTensor());
    }
    return res;
}
}  // namespace infer_detail

class DirectBackend : public InferenceBackend {
public:
    DirectBackend(torch::jit::script::Module module, torch::Device device, bool bf16 = false)
        : module_(std::move(module)), device_(device), bf16_(bf16 && device.is_cuda()) {
        module_.to(device_);
        module_.eval();
        has_oracle_ = module_.find_method("oracle").has_value();
    }

    InferOutputs Forward(const torch::Tensor& obs, const torch::Tensor& mask) override {
        torch::NoGradGuard g;
        AutocastGuard ac(bf16_);
        auto out = module_.forward({obs.to(device_), mask.to(device_)}).toTuple();
        return infer_detail::Unpack(out);
    }

    bool HasOracle() const override { return has_oracle_; }

    torch::Tensor OracleForward(const torch::Tensor& obs, const torch::Tensor& hands) override {
        torch::NoGradGuard g;
        AutocastGuard ac(bf16_);
        auto method = module_.find_method("oracle");
        return infer_detail::ToHost(
            (*method)({obs.to(device_), hands.to(device_)}).toTensor());
    }

private:
    torch::jit::script::Module module_;
    torch::Device device_;
    bool bf16_ = false;
    bool has_oracle_ = false;
};

class InferenceServer {
public:
    InferenceServer(torch::jit::script::Module module, torch::Device device, bool bf16 = false)
        : module_(std::move(module)), device_(device), bf16_(bf16 && device.is_cuda()) {
        module_.to(device_);
        module_.eval();
        has_oracle_ = module_.find_method("oracle").has_value();
        worker_ = std::thread([this] { Loop(); });
    }

    bool HasOracle() const { return has_oracle_; }

    ~InferenceServer() {
        {
            std::lock_guard<std::mutex> g(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        worker_.join();
    }

    InferOutputs Submit(const torch::Tensor& obs, const torch::Tensor& mask) {
        return Enqueue(obs, mask, false);
    }

    // Oracle result comes back in InferOutputs.value; logits/belief undefined.
    torch::Tensor SubmitOracle(const torch::Tensor& obs, const torch::Tensor& hands) {
        return Enqueue(obs, hands, true).value;
    }

    long long Launches() const { return launches_.load(); }
    double MeanBatchRows() const {
        long long l = launches_.load();
        return l ? static_cast<double>(rows_.load()) / l : 0.0;
    }

private:
    struct Request {
        torch::Tensor obs, aux;  // aux = mask (normal) or hands (oracle)
        bool is_oracle = false;
        std::promise<InferOutputs> promise;
    };

    InferOutputs Enqueue(const torch::Tensor& obs, const torch::Tensor& aux, bool is_oracle) {
        Request r;
        r.obs = obs;
        r.aux = aux;
        r.is_oracle = is_oracle;
        std::future<InferOutputs> fut = r.promise.get_future();
        {
            std::lock_guard<std::mutex> g(mu_);
            queue_.push_back(std::move(r));
        }
        cv_.notify_one();
        return fut.get();  // rethrows if the forward failed
    }

    void RunGroup(std::vector<Request*>& group, bool is_oracle) {
        if (group.empty()) return;
        torch::Tensor o, a;
        if (group.size() == 1) {
            o = group[0]->obs;
            a = group[0]->aux;
        } else {
            std::vector<torch::Tensor> os, as;
            os.reserve(group.size());
            as.reserve(group.size());
            for (const auto* r : group) {
                os.push_back(r->obs);
                as.push_back(r->aux);
            }
            o = torch::cat(os, 0);
            a = torch::cat(as, 0);
        }
        try {
            InferOutputs all;
            AutocastGuard ac(bf16_);
            if (is_oracle) {
                auto method = module_.find_method("oracle");
                all.value = infer_detail::ToHost(
                    (*method)({o.to(device_), a.to(device_)}).toTensor());
            } else {
                auto out = module_.forward({o.to(device_), a.to(device_)}).toTuple();
                all = infer_detail::Unpack(out);
            }
            launches_.fetch_add(1);
            rows_.fetch_add(o.size(0));
            int64_t row = 0;
            for (auto* r : group) {
                int64_t n = r->obs.size(0);
                InferOutputs res;
                if (is_oracle) {
                    res.value = all.value.slice(0, row, row + n);
                } else {
                    res.logits = all.logits.slice(0, row, row + n);
                    res.value = all.value.slice(0, row, row + n);
                    if (all.belief.defined()) {
                        res.belief = all.belief.slice(0, row, row + n);
                    }
                }
                r->promise.set_value(std::move(res));
                row += n;
            }
        } catch (...) {
            for (auto* r : group) {
                r->promise.set_exception(std::current_exception());
            }
        }
    }

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
            std::vector<Request*> normal, oracle;
            for (auto& r : batch) {
                (r.is_oracle ? oracle : normal).push_back(&r);
            }
            RunGroup(normal, false);
            RunGroup(oracle, true);
        }
    }

    torch::jit::script::Module module_;
    torch::Device device_;
    bool bf16_ = false;
    bool has_oracle_ = false;
    std::vector<Request> queue_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::thread worker_;
    // long long: Windows long is 32-bit and total rows overflow it
    std::atomic<long long> launches_{0};
    std::atomic<long long> rows_{0};
};

class ServedBackend : public InferenceBackend {
public:
    explicit ServedBackend(InferenceServer* server) : server_(server) {}

    InferOutputs Forward(const torch::Tensor& obs, const torch::Tensor& mask) override {
        return server_->Submit(obs, mask);
    }

    bool HasOracle() const override { return server_->HasOracle(); }

    torch::Tensor OracleForward(const torch::Tensor& obs, const torch::Tensor& hands) override {
        return server_->SubmitOracle(obs, hands);
    }

private:
    InferenceServer* server_;
};
