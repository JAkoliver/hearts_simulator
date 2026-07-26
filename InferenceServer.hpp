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
#include <deque>
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
#ifdef __linux__
// AOTInductor-compiled forward (HEARTS_SRV_AOTI=<pkg.pt2>): fused kernels
// from torch.export, compiled per-arch by cloud/export_aoti.py. Linux-only
// serving path (cloud); Windows keeps the JIT trace.
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#endif

// RAII guard: bf16 autocast for CUDA inference. The module stays fp32 (a
// traced module converted wholesale to bf16 crashes natively); autocast
// runs the matmul-heavy ops in bf16 the same way train.py does.
// Variant that leaves the autocast CAST CACHE alive across calls (the
// bf16 weight copies persist; ~15MB/model held instead of re-cast per
// forward). The per-call clear_cache() in AutocastGuard was measured as
// the 1.42x "persistent autocast" win on the server path and implicated
// in the 2026-07-25 8-shard driver wedge (per-call multi-MB alloc/free
// churn x 16 CUDA modules). Enable/disable still per-call so CPU-side
// code never runs under CUDA autocast.
class AutocastGuardPersistent {
public:
    explicit AutocastGuardPersistent(bool enable) : enabled_(enable) {
        if (enabled_) {
            at::autocast::set_autocast_enabled(at::kCUDA, true);
            at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
        }
    }
    ~AutocastGuardPersistent() {
        if (enabled_) {
            at::autocast::set_autocast_enabled(at::kCUDA, false);
        }
    }
private:
    bool enabled_;
};

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
#ifdef __linux__
        // Same env-gated AOTI path as InferenceServer, so SearchEval (which
        // is batch-1 through DirectBackend) exercises the SAME compiled
        // numerics the generation server uses - required for the R1 gate to
        // actually gate the AOTI stack.
        if (const char* pkg = std::getenv("HEARTS_SRV_AOTI")) {
            if (device_.is_cuda()) {
                try {
                    aoti_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(pkg);
                    std::fprintf(stderr, "[direct] AOTI package loaded: %s\n", pkg);
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "[direct] AOTI load failed (%s); JIT serves\n",
                                 e.what());
                }
            }
        }
#endif
    }

    // Round rows up to a bounded bucket set (1,2,4,...,512, then multiples
    // of 512): bounded distinct shapes -> caching-allocator and kernel-shape
    // stability. Padding rows are zeros; outputs are sliced back. Mirrors
    // the server's BucketRows fix - DirectBackend historically ran small
    // batches and never needed it until K=256 rollout batches.
    static int64_t BucketRowsDirect(int64_t n) {
        if (n <= 512) {
            int64_t b = 1;
            while (b < n) b <<= 1;
            return b;
        }
        return ((n + 511) / 512) * 512;
    }

    InferOutputs Forward(const torch::Tensor& obs, const torch::Tensor& mask) override {
        torch::NoGradGuard g;
#ifdef __linux__
        if (aoti_) {
            // bf16 baked into the compiled graph; no autocast needed
            auto outs = aoti_->run({obs.to(device_), mask.to(device_)});
            InferOutputs res;
            res.logits = infer_detail::ToHost(outs[0]);
            res.value = infer_detail::ToHost(outs[1]);
            if (outs.size() >= 3) res.belief = infer_detail::ToHost(outs[2]);
            return res;
        }
#endif
        const int64_t true_rows = obs.size(0);
        torch::Tensor o = obs, m = mask;
        if (device_.is_cuda()) {
            int64_t padded = BucketRowsDirect(true_rows);
            if (padded != true_rows) {
                o = torch::zeros({padded, obs.size(1)}, obs.options());
                o.narrow(0, 0, true_rows).copy_(obs);
                m = torch::zeros({padded, mask.size(1)}, mask.options());
                // Padding rows get a fully-true mask so softmax/argmax on
                // them stays finite; results are discarded by the slice.
                m.narrow(0, true_rows, padded - true_rows).fill_(true);
                m.narrow(0, 0, true_rows).copy_(mask);
            }
        }
        AutocastGuardPersistent ac(bf16_);
        auto out = module_.forward({o.to(device_), m.to(device_)}).toTuple();
        InferOutputs res = infer_detail::Unpack(out);
        if (res.logits.size(0) != true_rows) {
            res.logits = res.logits.narrow(0, 0, true_rows);
            if (res.value.defined()) res.value = res.value.narrow(0, 0, true_rows);
            if (res.belief.defined()) res.belief = res.belief.narrow(0, 0, true_rows);
        }
        return res;
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
#ifdef __linux__
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> aoti_;
#endif
};

class InferenceServer {
public:
    InferenceServer(torch::jit::script::Module module, torch::Device device, bool bf16 = false)
        : module_(std::move(module)), device_(device), bf16_(bf16 && device.is_cuda()) {
        module_.to(device_);
        module_.eval();
        has_oracle_ = module_.find_method("oracle").has_value();
#ifdef __linux__
        if (const char* pkg = std::getenv("HEARTS_SRV_AOTI")) {
            // bf16 is baked into the exported graph (see export_aoti.py);
            // outputs come back fp32. Oracle calls stay on the JIT module.
            try {
                aoti_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(pkg);
                std::fprintf(stderr, "[srv] AOTI package loaded: %s\n", pkg);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[srv] AOTI load failed (%s); JIT trace "
                                     "will serve instead\n", e.what());
            }
        }
#endif
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

    // ---- P1 pinned staging (default on CUDA; HEARTS_SRV_NOSTAGE=1 reverts) ----
    // Replaces the per-batch torch::cat + pageable-memory transfers with
    // reusable pinned host buffers per (bucket, method): requests memcpy
    // straight into the staging rows, H2D/D2H run async on the stream, and
    // one event sync replaces the blocking .to(kCPU). Kills the per-batch
    // allocation churn that inflated queue waits (measured 19-35 ms mean
    // wait on H100 vs 11-45 ms forwards).
    struct Staging {
        torch::Tensor h_obs, h_aux;        // pinned host inputs
        torch::Tensor d_obs, d_aux;        // persistent device inputs
        std::vector<torch::Tensor> h_out;  // pinned host outputs (lazy, f32)
    };

    Staging& GetStaging(int slot, int64_t bucket, bool is_oracle, int64_t obs_dim,
                        int64_t aux_dim, c10::ScalarType aux_type) {
        auto key = std::make_pair(bucket, is_oracle);
        auto& map = staging_[slot];
        auto it = map.find(key);
        if (it == map.end()) {
            Staging s;
            s.h_obs = torch::empty({bucket, obs_dim},
                torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
            s.h_aux = torch::empty({bucket, aux_dim},
                torch::TensorOptions().dtype(aux_type).pinned_memory(true));
            s.d_obs = torch::empty({bucket, obs_dim},
                torch::TensorOptions().dtype(torch::kFloat32).device(device_));
            s.d_aux = torch::empty({bucket, aux_dim},
                torch::TensorOptions().dtype(aux_type).device(device_));
            it = map.emplace(key, std::move(s)).first;
        }
        return it->second;
    }

    void RunGroupStaged(std::vector<Request*>& group, bool is_oracle) {
        int64_t true_rows = 0;
        for (const auto* r : group) true_rows += r->obs.size(0);
        int64_t padded = BucketRows(true_rows);
        Staging& st = GetStaging(0, padded, is_oracle, group[0]->obs.size(1),
                                 group[0]->aux.size(1),
                                 group[0]->aux.scalar_type());
        int64_t row = 0;
        for (const auto* r : group) {
            int64_t n = r->obs.size(0);
            st.h_obs.slice(0, row, row + n).copy_(r->obs);
            st.h_aux.slice(0, row, row + n).copy_(r->aux);
            row += n;
        }
        if (padded > true_rows) {
            st.h_obs.slice(0, true_rows, padded).zero_();
            if (is_oracle) {
                st.h_aux.slice(0, true_rows, padded).zero_();
            } else {
                st.h_aux.slice(0, true_rows, padded).fill_(1);  // all-legal dummies
            }
        }
        try {
            auto t0 = std::chrono::steady_clock::now();
            st.d_obs.copy_(st.h_obs, /*non_blocking=*/true);
            st.d_aux.copy_(st.h_aux, /*non_blocking=*/true);
            std::vector<torch::Tensor> outs;
#ifdef __linux__
            if (aoti_ && !is_oracle) {
                outs = aoti_->run({st.d_obs, st.d_aux});
            } else
#endif
            if (is_oracle) {
                auto method = module_.find_method("oracle");
                outs.push_back((*method)({st.d_obs, st.d_aux}).toTensor());
            } else {
                auto out = module_.forward({st.d_obs, st.d_aux}).toTuple();
                for (const auto& v : out->elements()) outs.push_back(v.toTensor());
            }
            if (st.h_out.size() != outs.size()) {
                st.h_out.clear();
                for (const auto& o : outs) {
                    st.h_out.push_back(torch::empty(o.sizes(),
                        torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true)));
                }
            }
            // D2H (with bf16->f32 conversion) into pinned buffers; one event
            // sync instead of a blocking .to(kCPU) per output
            for (size_t i = 0; i < outs.size(); ++i) {
                st.h_out[i].copy_(outs[i], /*non_blocking=*/true);
            }
            // One stream sync covers H2D + forward + all D2H copies (all on
            // this thread's stream) - replaces a blocking .to(kCPU) per output
            c10::cuda::getCurrentCUDAStream().synchronize();
            launches_.fetch_add(1);
            rows_.fetch_add(padded);
            if (perf_on_) {
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
            row = 0;
            for (auto* r : group) {
                int64_t n = r->obs.size(0);
                InferOutputs res;
                // clone: the pinned buffers are reused by the next batch
                if (is_oracle) {
                    res.value = st.h_out[0].slice(0, row, row + n).clone();
                } else {
                    res.logits = st.h_out[0].slice(0, row, row + n).clone();
                    res.value = st.h_out[1].slice(0, row, row + n).clone();
                    if (st.h_out.size() >= 3) {
                        res.belief = st.h_out[2].slice(0, row, row + n).clone();
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

    // ---- P2 two-slot pipeline (HEARTS_SRV_PIPE=1) ----
    // While slot A's kernels run, slot B's batch is assembled on the CPU and
    // its async chain launched on a second stream; retirement (sync +
    // fulfill) of A then usually finds the GPU already done. Recovers the
    // CPU gaps BETWEEN forwards; the in-forward queueing wait is intrinsic
    // to a saturated GPU and stays.
    struct Pipe {
        std::vector<Request> owned;
        std::vector<Request*> group;
        bool is_oracle = false;
        int64_t true_rows = 0, padded = 0;
        Staging* st = nullptr;
        std::chrono::steady_clock::time_point t0;
        bool active = false;
    };

    void LaunchPipe(Pipe& p, int slot) {
        p.true_rows = 0;
        for (const auto* r : p.group) p.true_rows += r->obs.size(0);
        p.padded = BucketRows(p.true_rows);
        p.st = &GetStaging(slot, p.padded, p.is_oracle, p.group[0]->obs.size(1),
                           p.group[0]->aux.size(1),
                           p.group[0]->aux.scalar_type());
        Staging& st = *p.st;
        int64_t row = 0;
        for (const auto* r : p.group) {
            int64_t n = r->obs.size(0);
            st.h_obs.slice(0, row, row + n).copy_(r->obs);
            st.h_aux.slice(0, row, row + n).copy_(r->aux);
            row += n;
        }
        if (p.padded > p.true_rows) {
            st.h_obs.slice(0, p.true_rows, p.padded).zero_();
            if (p.is_oracle) st.h_aux.slice(0, p.true_rows, p.padded).zero_();
            else st.h_aux.slice(0, p.true_rows, p.padded).fill_(1);
        }
        try {
            p.t0 = std::chrono::steady_clock::now();
            st.d_obs.copy_(st.h_obs, /*non_blocking=*/true);
            st.d_aux.copy_(st.h_aux, /*non_blocking=*/true);
            std::vector<torch::Tensor> outs;
#ifdef __linux__
            if (aoti_ && !p.is_oracle) {
                outs = aoti_->run({st.d_obs, st.d_aux});
            } else
#endif
            if (p.is_oracle) {
                auto method = module_.find_method("oracle");
                outs.push_back((*method)({st.d_obs, st.d_aux}).toTensor());
            } else {
                auto out = module_.forward({st.d_obs, st.d_aux}).toTuple();
                for (const auto& v : out->elements()) outs.push_back(v.toTensor());
            }
            if (st.h_out.size() != outs.size()) {
                st.h_out.clear();
                for (const auto& o : outs) {
                    st.h_out.push_back(torch::empty(o.sizes(),
                        torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true)));
                }
            }
            for (size_t i = 0; i < outs.size(); ++i) {
                st.h_out[i].copy_(outs[i], /*non_blocking=*/true);
            }
            p.active = true;
        } catch (...) {
            for (auto* r : p.group) {
                r->promise.set_exception(std::current_exception());
            }
            p.owned.clear();
            p.group.clear();
            p.active = false;
        }
    }

    void RetirePipe(Pipe& p) {
        try {
            c10::cuda::getCurrentCUDAStream().synchronize();
            launches_.fetch_add(1);
            rows_.fetch_add(p.padded);
            if (perf_on_) {
                auto t1 = std::chrono::steady_clock::now();
                auto& slot = perf_.fwd[p.padded];
                slot.first += 1;
                slot.second += std::chrono::duration<double, std::milli>(t1 - p.t0).count();
                perf_.true_rows += p.true_rows;
                perf_.pad_rows += p.padded;
                for (const auto* r : p.group) {
                    perf_.wait_ms +=
                        std::chrono::duration<double, std::milli>(p.t0 - r->t_enq).count();
                    perf_.n += 1;
                }
                if (launches_.load() % 500 == 0) perf_.Report(launches_.load());
            }
            int64_t row = 0;
            Staging& st = *p.st;
            for (auto* r : p.group) {
                int64_t n = r->obs.size(0);
                InferOutputs res;
                if (p.is_oracle) {
                    res.value = st.h_out[0].slice(0, row, row + n).clone();
                } else {
                    res.logits = st.h_out[0].slice(0, row, row + n).clone();
                    res.value = st.h_out[1].slice(0, row, row + n).clone();
                    if (st.h_out.size() >= 3) {
                        res.belief = st.h_out[2].slice(0, row, row + n).clone();
                    }
                }
                r->promise.set_value(std::move(res));
                row += n;
            }
        } catch (...) {
            for (auto* r : p.group) {
                r->promise.set_exception(std::current_exception());
            }
        }
        p.owned.clear();
        p.group.clear();
        p.active = false;
    }

    void LoopPipelined() {
        c10::cuda::CUDAStream streams[2] = {
            c10::cuda::getStreamFromPool(false, device_.index()),
            c10::cuda::getStreamFromPool(false, device_.index())};
        Pipe pipes[2];
        struct Unit { std::vector<Request> owned; bool is_oracle; };
        std::deque<Unit> pending;
        int oldest = -1;
        int n_active = 0;
        while (true) {
            std::vector<Request> batch;
            {
                std::unique_lock<std::mutex> lk(mu_);
                if (queue_.empty() && pending.empty() && n_active == 0) {
                    cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
                }
                if (stop_ && queue_.empty() && pending.empty() && n_active == 0) {
                    return;
                }
                batch.swap(queue_);
            }
            if (!batch.empty()) {
                std::vector<Request> normal, oracle;
                for (auto& r : batch) {
                    (r.is_oracle ? oracle : normal).push_back(std::move(r));
                }
                if (!normal.empty()) pending.push_back({std::move(normal), false});
                if (!oracle.empty()) pending.push_back({std::move(oracle), true});
            }
            if (!pending.empty() && n_active < 2) {
                int slot = (n_active == 0) ? 0 : (oldest ^ 1);
                Pipe& p = pipes[slot];
                p.owned = std::move(pending.front().owned);
                p.is_oracle = pending.front().is_oracle;
                pending.pop_front();
                p.group.clear();
                for (auto& r : p.owned) p.group.push_back(&r);
                c10::cuda::setCurrentCUDAStream(streams[slot]);
                LaunchPipe(p, slot);
                if (p.active) {
                    if (n_active == 0) oldest = slot;
                    ++n_active;
                }
                continue;  // try to fill the second slot before retiring
            }
            if (n_active > 0) {
                int slot = oldest;
                c10::cuda::setCurrentCUDAStream(streams[slot]);
                RetirePipe(pipes[slot]);
                --n_active;
                oldest = (n_active > 0) ? (slot ^ 1) : -1;
            }
        }
    }

    void RunGroup(std::vector<Request*>& group, bool is_oracle) {
        if (group.empty()) return;
        if (staged_on_ && device_.is_cuda() && !graphs_on_) {
            RunGroupStaged(group, is_oracle);
            return;
        }
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
#ifdef __linux__
            if (aoti_ && !is_oracle) {
                auto outs = aoti_->run({o.to(device_), a.to(device_)});
                all.logits = infer_detail::ToHost(outs[0]);
                all.value = infer_detail::ToHost(outs[1]);
                if (outs.size() >= 3) all.belief = infer_detail::ToHost(outs[2]);
                ran = true;
            }
#endif
            if (!ran && graphs_on_ && device_.is_cuda() && padded <= 8192) {
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
        if (pipe_on_ && device_.is_cuda() && !graphs_on_) {
            LoopPipelined();
            return;
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
    // P1 measured neutral on the 4090 (319s == 319s, waits unchanged):
    // pinned copies don't touch the dominant wait source, which is
    // queueing behind in-flight forwards. Opt-in only; P2's pipeline
    // (HEARTS_SRV_PIPE=1) builds on it and is where overlap happens.
    bool staged_on_ = std::getenv("HEARTS_SRV_STAGE") != nullptr;
    bool pipe_on_ = std::getenv("HEARTS_SRV_PIPE") != nullptr;
    std::map<std::pair<int64_t, bool>, Staging> staging_[2];
#ifdef __linux__
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> aoti_;
#endif
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
