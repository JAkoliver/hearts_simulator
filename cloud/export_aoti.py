"""AOTInductor export of the v5 search model (forward only).

Compiles the policy+value+belief forward into a fused-kernel .pt2 package
for the GPU present at compile time (Inductor autotunes per-arch: run this
ON the card that will serve it - 4090/sm_89 locally, H100/sm_90 on the
pod). The oracle head stays on the JIT trace (unused by production
generation and the gate).

bf16 is baked into the exported graph via autocast inside the wrapper, so
the C++ side runs the package WITHOUT an autocast guard - numerics match
the JIT+autocast path to bf16 rounding.

Batch dimension is exported dynamic: one package serves every bucket size.

Usage (inside the hearts-aoti container, GPU visible):
  python3 cloud/export_aoti.py --checkpoint hearts_model_final.pth \
      --out hearts_ai_search_aoti.pt2
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hearts_net import net_from_checkpoint


class SearchForwardBF16(torch.nn.Module):
    """forward_all under bf16 autocast, cast back to fp32 - the same
    numerics contract the C++ server exposes today (bf16 compute, fp32
    outputs)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, observation, legal_actions_mask):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, value, belief = self.net.forward_all(
                observation, legal_actions_mask)
        return logits.float(), value.float(), belief.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='hearts_model_final.pth')
    ap.add_argument('--out', default='hearts_ai_search_aoti.pt2')
    ap.add_argument('--max-batch', type=int, default=8192)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "compile on the GPU that will serve this"
    dev = torch.device('cuda')
    print(f"compiling on: {torch.cuda.get_device_name(0)}")

    net = net_from_checkpoint(args.checkpoint).to(dev).eval()
    wrapper = SearchForwardBF16(net).eval()

    B = torch.export.Dim('batch', min=1, max=args.max_batch)
    ex_obs = torch.rand(64, 550, device=dev)
    ex_mask = torch.rand(64, 52, device=dev) > 0.3
    ex_mask[:, 0] = True

    with torch.no_grad():
        ep = torch.export.export(
            wrapper, (ex_obs, ex_mask),
            dynamic_shapes={'observation': {0: B}, 'legal_actions_mask': {0: B}})
        pkg = torch._inductor.aoti_compile_and_package(
            ep, package_path=args.out,
            inductor_configs={'max_autotune': True})
    print(f"packaged: {pkg}")

    # Parity check vs eager, plus a quick timing vs the JIT trace if present
    from torch._inductor.package import load_package
    compiled = load_package(args.out)
    for rows in (64, 1536, 4096):
        obs = torch.rand(rows, 550, device=dev)
        mask = torch.rand(rows, 52, device=dev) > 0.3
        mask[:, 0] = True
        with torch.no_grad():
            cl, cv, cb = compiled(obs, mask)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                el, ev, eb = net.forward_all(obs, mask)
        legal = mask
        dl = (cl - el.float())[legal].abs().max().item()
        agree = (cl.argmax(1) == el.float().argmax(1)).float().mean().item()
        print(f"rows {rows}: max|dlogits| {dl:.3e}  argmax agree {agree*100:.2f}%")

    def bench(fn, obs, mask, iters=50):
        torch.cuda.synchronize()
        import time
        t0 = time.time()
        for _ in range(iters):
            with torch.no_grad():
                fn(obs, mask)
        torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1000

    obs = torch.rand(2048, 550, device=dev)
    mask = torch.rand(2048, 52, device=dev) > 0.3
    mask[:, 0] = True
    for _ in range(10):  # warmup both paths
        with torch.no_grad():
            compiled(obs, mask)
    t_aoti = bench(compiled, obs, mask)
    print(f"AOTI 2048 rows: {t_aoti:.2f} ms/launch")
    if os.path.exists('hearts_ai_search.pt'):
        jit = torch.jit.load('hearts_ai_search.pt').to(dev).eval()
        def jit_fn(o, m):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return jit(o, m)
        for _ in range(10):
            with torch.no_grad():
                jit_fn(obs, mask)
        t_jit = bench(jit_fn, obs, mask)
        print(f"JIT  2048 rows: {t_jit:.2f} ms/launch  ->  AOTI speedup {t_jit / t_aoti:.2f}x")


if __name__ == '__main__':
    main()
