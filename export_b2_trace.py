"""Export a B2 CANDIDATE's 556-dim match trace for the defender slot.

Unlike export_match.py (hardcoded to hearts_model_final.pth and
OVERWRITES the frozen instrument traces hearts_ai_match.pt /
hearts_ai_search_match.pt), this takes explicit paths and refuses to
write onto any frozen instrument file. The output is what SearchEval's
--opponent-model consumes for defender seats (same trace surface as the
frozen base defender trace: forward(obs556, mask52) -> logits).

Verifies trace parity: argmax agreement with the source .pth on 2,048
real held-out positions must be exact (bit-level trace faithfulness on
the same device).

Usage: python export_b2_trace.py <candidate.pth> <out_trace.pt>
"""
import argparse
import sys

import numpy as np
import torch

from hearts_net import net_from_checkpoint

FROZEN = {'hearts_ai_match.pt', 'hearts_ai_search_match.pt',
          'hearts_ai_search.pt', 'hearts_equity.pt'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('out')
    args = ap.parse_args()
    if args.out.replace('\\', '/').split('/')[-1] in FROZEN:
        sys.exit(f'refusing to overwrite frozen instrument trace {args.out}')

    net = net_from_checkpoint(args.ckpt)
    net.eval()
    traced = torch.jit.trace(net, (torch.zeros(1, 556),
                                   torch.zeros(1, 52, dtype=torch.bool)))
    # sanity: 556 accepted, 550 rejected (match_proj baked in)
    traced(torch.rand(2, 556), torch.ones(2, 52, dtype=torch.bool))
    try:
        traced(torch.rand(2, 550), torch.ones(2, 52, dtype=torch.bool))
        sys.exit('ERROR: trace accepted 550-dim input')
    except RuntimeError:
        pass

    # parity on real positions (CPU, same device for both sides)
    z = np.load('b2_data/drift_holdout.npz')
    n = min(2048, len(z['obs']))
    obs = torch.from_numpy(np.ascontiguousarray(z['obs'][:n]))
    mask = torch.from_numpy(np.ascontiguousarray(z['mask'][:n])).bool()
    with torch.no_grad():
        a1 = net.forward_all(obs, mask)[0].argmax(1)
        a2 = traced(obs, mask)
        a2 = (a2[0] if isinstance(a2, tuple) else a2).argmax(1)
    if not torch.equal(a1, a2):
        sys.exit(f'ERROR: trace parity failed '
                 f'({int((a1 != a2).sum())}/{n} argmax flips)')
    traced.save(args.out)
    print(f'{args.out}: exported, 550-reject OK, parity exact on {n} positions')


if __name__ == '__main__':
    main()
