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
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
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
        if (std::getenv("HEARTS_SRV_OFI") != nullptr) {
            // Measured on the v5 transformer (July 2026): ~17% SLOWER per
            // launch under bf16 autocast (freezing weights to constants
            // defeats the autocast cast cache). Kept env-gated for future
            // re-measurement on other models; do not enable by default.
            try {
                std::vector<std::string> keep;
                if (has_oracle_) keep.push_back("oracle");
                auto frozen = torch::jit::freeze(module_, keep);
                module_ = torch::jit::optimize_for_inference(frozen, keep);
                std::fprintf(stderr, "[srv] optimize_for_inference applied\n");
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[srv] optimize_for_inference failed (%s); "
                                     "using the plain module\n", e.what());
            }
        }
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
        std::chrono::steady_clock::time_point t_enq;
        std::promise<InferOutputs> promise;
    };

    // HEARTS_SRV_PERF=1: windowed per-bucket forward timings to stderr.
    // Diagnostic for the process-age throughput decay - if ms/launch at a
    // FIXED bucket size grows over the run, the degradation is inside the
    // forward (JIT executor / allocator), not in the callers.
    struct PerfWindow {
        // bucket rows -> (launches, total forward ms); touched only by the
        // server thread, no locking needed
        std::map<int64_t, std::pair<long, double>> fwd;
        double wait_ms = 0.0;
        long n = 0;
        long long true_rows = 0, pad_rows = 0;
        std::chrono::steady_clock::time_point window_t0 =
            std::chrono::steady_clock::now();

        void Report(long long total_launches) {
            auto now = std::chrono::steady_clock::now();
            double win_s = std::chrono::duration<double>(now - window_t0).count();
            std::fprintf(stderr,
                         "[srv] launch %lld  window %.0fs  wait %.2fms  pad %.1f%% |",
                         total_launches, win_s, n ? wait_ms / n : 0.0,
                         pad_rows ? 100.0 * (pad_rows - true_rows) / pad_rows : 0.0);
            for (const auto& kv : fwd) {
                std::fprintf(stderr, "  %lld: %.2fms x%ld",
                             static_cast<long long>(kv.first),
                             kv.second.first ? kv.second.second / kv.second.first : 0.0,
                             kv.second.first);
            }
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
            fwd.clear();
            wait_ms = 0.0;
            n = 0;
            true_rows = 0;
            pad_rows = 0;
            window_t0 = now;
        }
    };

    InferOutputs Enqueue(const torch::Tensor& obs, const torch::Tensor& aux, bool is_oracle) {
        Request r;
        r.obs = obs;
        r.aux = aux;
        r.is_oracle = is_oracle;
        r.t_enq = std::chrono::steady_clock::now();
        std::future<InferOutputs> fut = r.promise.get_future();
        {
            std::lock_guard<std::mutex> g(mu_);
            queue_.push_back(std::move(r));
        }
        cv_.notify_one();
        return fut.get();  // rethrows if the forward failed
    }

    // Round row counts up to a small fixed set of bucket sizes. The server
    // otherwise produces thousands of DISTINCT batch shapes, and every
    // shape-keyed cache in the stack (JIT plan specialization, CUDA
    // allocator size classes) grows without bound - measured on the v5
    // transformer as stepwise VRAM growth to 24 GB and a 2 -> 15 s/deal
    // decay. Padding rows are zeros; results are sliced back to true rows.
    int64_t BucketRows(int64_t n) const {
        if (graphs_on_) {
            // Graph path: fewest possible shapes - every captured graph owns
            // a PRIVATE memory pool (shared pools require replaying in
            // capture order, which the server can't guarantee), and padding
            // compute is nearly free once the launches are fused.
            int64_t b = 256;
            while (b < n) b *= 2;
            return b;
        }
        int64_t b = 32;
        while (b < n && b < 512) b *= 2;
        if (n <= b) return b;
        // Multiples of 512 above 512: mean padding waste ~10% instead of the
        // ~33% of pure powers of 2, while the shape set stays small enough
        // (~20 sizes) that shape-keyed caches remain bounded.
        return (n + 511) / 512 * 512;
    }

    // ---- CUDA Graph replay (HEARTS_SRV_GRAPH=1) ----
    // The v5 forward is launch-overhead-bound: hundreds of tiny JIT-
    // dispatched kernels put the 4090 at ~1% of its bf16 peak (22 us/row).
    // With batch shapes drawn from a small fixed set, each (rows, method)
    // pair is captured once and replayed thereafter, skipping all per-op
    // CPU dispatch and per-kernel launch latency.
    struct GraphEntry {
        at::cuda::CUDAGraph graph;
        torch::Tensor obs, aux;           // static device-side inputs
        std::vector<torch::Tensor> outs;  // static outputs (graph pool)
        bool ok = false;
    };

    std::unique_ptr<GraphEntry> CaptureGraph(bool is_oracle,
                                             const torch::Tensor& o,
                                             const torch::Tensor& a) {
        auto e = std::make_unique<GraphEntry>();
        // Bake the fp32->bf16 weight casts INTO the graph: with the autocast
        // cast cache enabled, the captured graph would hold pointers into a
        // cache that any outside clear_cache() could free (use-after-free on
        // replay). Self-contained costs well under a millisecond per replay.
        at::autocast::set_autocast_cache_enabled(false);
        at::autocast::clear_cache();
        try {
            e->obs = o.to(device_);
            e->aux = a.to(device_);
            auto run = [&]() {
                std::vector<torch::Tensor> outs;
                if (is_oracle) {
                    auto method = module_.find_method("oracle");
                    outs.push_back((*method)({e->obs, e->aux}).toTensor());
                } else {
                    auto out = module_.forward({e->obs, e->aux}).toTuple();
                    for (const auto& v : out->elements()) {
                        outs.push_back(v.toTensor());
                    }
                }
                return outs;
            };
            // Settle JIT plan specialization and kernel autotuning for this
            // shape before recording
            for (int i = 0; i < 3; ++i) run();
            c10::cuda::getCurrentCUDAStream().synchronize();
            e->graph.capture_begin();
            e->outs = run();
            e->graph.capture_end();
            e->graph.replay();  // validate before declaring the graph usable
            c10::cuda::getCurrentCUDAStream().synchronize();
            e->ok = true;
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[srv] cuda graph capture failed (rows %lld%s): %s\n",
                         static_cast<long long>(o.size(0)),
                         is_oracle ? ", oracle" : "", ex.what());
            e->ok = false;
        }
        at::autocast::set_autocast_cache_enabled(true);
        return e;
    }

    // Returns false if this shape must run eagerly (capture failed / too big)
    bool ForwardGraph(int64_t padded, bool is_oracle, const torch::Tensor& o,
                      const torch::Tensor& a, InferOutputs& all) {
        auto key = std::make_pair(padded, is_oracle);
        auto it = graphs_.find(key);
        if (it == graphs_.end()) {
            it = graphs_.emplace(key, CaptureGraph(is_oracle, o, a)).first;
            if (it->second->ok) {
                std::fprintf(stderr, "[srv] cuda graph ready: %lld rows%s\n",
                             static_cast<long long>(padded),
                             is_oracle ? " (oracle)" : "");
            }
        }
        GraphEntry& g = *it->second;
        if (!g.ok) return false;
        g.obs.copy_(o);
        g.aux.copy_(a);
        g.graph.replay();
        if (is_oracle) {
            all.value = infer_detail::ToHost(g.outs[0]);
        } else {
            all.logits = infer_detail::ToHost(g.outs[0]);
            all.value = infer_detail::ToHost(g.outs[1]);
            if (g.outs.size() >= 3) {
                all.belief = infer_detail::ToHost(g.outs[2]);
            }
        }
        return true;
    }

    void RunGroup(std::vector<Request*>& group, bool is_oracle) {
        if (group.empty()) return;
        int64_t true_rows = 0;
        for (const auto* r : group) true_rows += r->obs.size(0);
        int64_t padded = BucketRows(true_rows);

        std::vector<torch::Tensor> os, as;
        os.reserve(group.size() + 1);
        as.reserve(group.size() + 1);
        for (const auto* r : group) {
            os.push_back(r->obs);
            as.push_back(r->aux);
        }
        if (padded > true_rows) {
            os.push_back(torch::zeros({padded - true_rows, os[0].size(1)},
                                      os[0].options()));
            as.push_back(is_oracle
                             ? torch::zeros({padded - true_rows, as[0].size(1)},
                                            as[0].options())
                             : torch::ones({padded - true_rows, as[0].size(1)},
                                           as[0].options()));  // masks: all-legal dummies
        }
        torch::Tensor o = torch::cat(os, 0);
        torch::Tensor a = torch::cat(as, 0);
        try {
            auto t0 = std::chrono::steady_clock::now();
            InferOutputs all;
            bool ran = false;
            if (graphs_on_ && device_.is_cuda() && padded <= 8192) {
                ran = ForwardGraph(padded, is_oracle, o, a, all);
            }
            if (!ran) {
                if (is_oracle) {
                    auto method = module_.find_method("oracle");
                    all.value = infer_detail::ToHost(
                        (*method)({o.to(device_), a.to(device_)}).toTensor());
                } else {
                    auto out = module_.forward({o.to(device_), a.to(device_)}).toTuple();
                    all = infer_detail::Unpack(out);
                }
            }
            launches_.fetch_add(1);
            rows_.fetch_add(o.size(0));
            if (perf_on_) {
                // ToHost's .to(kCPU) synchronizes, so t1 - t0 is the true
                // wall cost of this launch (H2D + forward + D2H)
                auto t1 = std::chrono::steady_clock::now();
                auto& slot = perf_.fwd[padded];
                slot.first += 1;
                slot.second += std::chrono::duration<double, std::milli>(t1 - t0).count();
                perf_.true_rows += true_rows;
                perf_.pad_rows += padded;
                for (const auto* r : group) {
                    perf_.wait_ms +=
                        std::chrono::duration<double, std::milli>(t0 - r->t_enq).count();
                    perf_.n += 1;
                }
                if (launches_.load() % 500 == 0) perf_.Report(launches_.load());
            }
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
        // Autocast held for the server thread's whole life (all forwards run
        // here): the bf16 weight-cast cache then persists across launches
        // instead of re-casting every parameter on every batch, which a
        // per-launch guard forced via clear_cache(). Cache footprint is one
        // bf16 copy of the weights (~15 MB) - bounded.
        AutocastGuard ac(bf16_);
        // Graph capture is illegal on the default stream; give the server
        // thread its own stream for everything (warmup, capture, replay,
        // eager fallback) so all work stays consistently ordered.
        if (graphs_on_ && device_.is_cuda()) {
            c10::cuda::setCurrentCUDAStream(
                c10::cuda::getStreamFromPool(false, device_.index()));
        }
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
    bool perf_on_ = std::getenv("HEARTS_SRV_PERF") != nullptr;
    bool graphs_on_ = std::getenv("HEARTS_SRV_GRAPH") != nullptr;
    PerfWindow perf_;
    std::map<std::pair<int64_t, bool>, std::unique_ptr<GraphEntry>> graphs_;
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
