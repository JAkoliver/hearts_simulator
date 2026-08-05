"""Phase 1 of client-side search (hearts_web/TODO.md): export the deployed
policy (556-dim match trace surface, forward_all: logits/value/belief) and
the equity net to ONNX for in-browser inference (onnxruntime-web), with a
parity harness against the DEPLOYED TorchScript traces.

Outputs (gitignored - the repo carries no weights):
    hearts_web/static/models/perilune_policy.onnx
    hearts_web/static/models/perilune_equity.onnx
    hearts_web/static/models/manifest.json   (md5s, source hashes, opset)
"""
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn

from hearts_net import net_from_checkpoint
from train_equity import EquityNet

OUT_DIR = os.path.join('hearts_web', 'static', 'models')
CKPT = 'hearts_model_final.pth'
OPSET = 17


class PolicyExport(nn.Module):
    """forward_all surface plus an argmax head: rollout rounds fetch only
    'act' (8 bytes/row instead of a 208-byte logits row - GPU->JS readback
    is part of the WebGPU round cost)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, observation, legal_actions_mask):
        logits, value, belief = self.net.forward_all(observation,
                                                     legal_actions_mask)
        return logits, value, belief, logits.argmax(1)


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ck_md5 = md5(CKPT)
    print(f"checkpoint {CKPT} md5 {ck_md5[:8]} (expect baseline 8a89da90)")

    net = net_from_checkpoint(CKPT)
    net.eval()
    policy = PolicyExport(net).eval()

    obs = torch.zeros(2, 556)
    mask = torch.ones(2, 52, dtype=torch.bool)
    ppath = os.path.join(OUT_DIR, 'perilune_policy.onnx')
    torch.onnx.export(
        policy, (obs, mask), ppath, opset_version=OPSET,
        input_names=['obs', 'mask'],
        output_names=['logits', 'value', 'belief', 'act'],
        dynamic_axes={'obs': {0: 'batch'}, 'mask': {0: 'batch'},
                      'logits': {0: 'batch'}, 'value': {0: 'batch'},
                      'belief': {0: 'batch'}, 'act': {0: 'batch'}},
        dynamo=False)
    print(f"exported {ppath} ({os.path.getsize(ppath) / 1e6:.1f} MB)")

    # fp16 variant for WebGPU (bandwidth-bound: ~2x; io kept fp32 so the
    # JS side never handles half floats - casts live in the graph).
    # Exported natively from a .half() model rather than post-converted:
    # the converter mangles the SDPA-region Casts. bf16 CUDA precedent
    # says the search tolerates reduced precision; the flip rate below
    # quantifies it.
    class PolicyExportFP16(nn.Module):
        def __init__(self, net16):
            super().__init__()
            self.net = net16

        def forward(self, observation, legal_actions_mask):
            logits, value, belief = self.net.forward_all(
                observation.half(), legal_actions_mask)
            return (logits.float(), value.float(), belief.float(),
                    logits.argmax(1))

    net16 = net_from_checkpoint(CKPT).half().eval()
    p16path = os.path.join(OUT_DIR, 'perilune_policy_fp16.onnx')
    torch.onnx.export(
        PolicyExportFP16(net16), (obs, mask), p16path, opset_version=OPSET,
        input_names=['obs', 'mask'],
        output_names=['logits', 'value', 'belief', 'act'],
        dynamic_axes={'obs': {0: 'batch'}, 'mask': {0: 'batch'},
                      'logits': {0: 'batch'}, 'value': {0: 'batch'},
                      'belief': {0: 'batch'}, 'act': {0: 'batch'}},
        dynamo=False)
    print(f"exported {p16path} ({os.path.getsize(p16path) / 1e6:.1f} MB)")

    ck = torch.load('equity_v1.pth', weights_only=True)
    eq = EquityNet(ck['in_dim'])
    eq.load_state_dict(ck['state_dict'])
    eq.eval()
    epath = os.path.join(OUT_DIR, 'perilune_equity.onnx')
    torch.onnx.export(
        eq, (torch.zeros(2, 10),), epath, opset_version=OPSET,
        input_names=['x'], output_names=['logits'],
        dynamic_axes={'x': {0: 'batch'}, 'logits': {0: 'batch'}},
        dynamo=False)
    print(f"exported {epath} ({os.path.getsize(epath) / 1e6:.3f} MB)")

    # ---- parity vs the DEPLOYED TorchScript traces --------------------------
    import onnxruntime as ort
    ts_policy = torch.jit.load('hearts_ai_search_match.pt').eval()
    ts_equity = torch.jit.load('hearts_equity.pt').eval()
    sp = ort.InferenceSession(ppath, providers=['CPUExecutionProvider'])
    se = ort.InferenceSession(epath, providers=['CPUExecutionProvider'])

    s16 = ort.InferenceSession(p16path, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(0)
    worst = {'logits': 0.0, 'value': 0.0, 'belief': 0.0, 'equity': 0.0,
             'argmax_mismatch': 0, 'act_vs_logits_mismatch': 0,
             'fp16_argmax_flips': 0, 'fp16_rows': 0}
    for trial in range(20):
        b = int(rng.integers(1, 96))
        o = rng.random((b, 556), dtype=np.float32)
        m = np.zeros((b, 52), dtype=bool)
        for i in range(b):                       # realistic ragged legal sets
            m[i, rng.choice(52, size=int(rng.integers(1, 14)),
                            replace=False)] = True
        with torch.no_grad():
            tl, tv, tb = ts_policy(torch.from_numpy(o), torch.from_numpy(m))
        ol, ov, ob, oa = sp.run(None, {'obs': o, 'mask': m})
        worst['act_vs_logits_mismatch'] += int(
            (oa != np.where(m, ol, -np.inf).argmax(1)).sum())
        oa16 = s16.run(['act'], {'obs': o, 'mask': m})[0]
        worst['fp16_argmax_flips'] += int((oa16 != oa).sum())
        worst['fp16_rows'] += b
        lm = np.abs(np.where(m, ol - tl.numpy(), 0.0)).max()
        worst['logits'] = max(worst['logits'], float(lm))
        worst['value'] = max(worst['value'], float(np.abs(ov - tv.numpy()).max()))
        worst['belief'] = max(worst['belief'], float(np.abs(ob - tb.numpy()).max()))
        tmask = np.where(m, tl.numpy(), -np.inf).argmax(1)
        omask = np.where(m, ol, -np.inf).argmax(1)
        worst['argmax_mismatch'] += int((tmask != omask).sum())

        x = rng.random((b, 10), dtype=np.float32)
        with torch.no_grad():
            te = ts_equity(torch.from_numpy(x))
        oe = se.run(None, {'x': x})[0]
        worst['equity'] = max(worst['equity'], float(np.abs(oe - te.numpy()).max()))

    print("parity (max abs diff vs TorchScript traces, 20 random batches):")
    for k, v in worst.items():
        print(f"  {k}: {v}")
    ok = (worst['logits'] < 1e-3 and worst['belief'] < 1e-3
          and worst['equity'] < 1e-4 and worst['argmax_mismatch'] == 0)
    print("PARITY", "PASS" if ok else "FAIL")

    with open(os.path.join(OUT_DIR, 'manifest.json'), 'w') as f:
        json.dump({'checkpoint_md5': ck_md5, 'opset': OPSET,
                   'policy_onnx_md5': md5(ppath), 'equity_onnx_md5': md5(epath),
                   'parity': worst, 'parity_pass': ok}, f, indent=2)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
