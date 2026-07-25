"""Export the 556-dim MATCH trace of the current baseline.

Traced with a 556-dim dummy so the match_proj branch is baked in: this
trace REQUIRES 556-dim input (engine obs + 6 match-ctx dims) and is what
SearchEval's --match-model mode consumes. Not produced by export.py -
regenerate after promotions when running match-bridge measurements.
"""
import torch

from hearts_net import net_from_checkpoint

net = net_from_checkpoint('hearts_model_final.pth')
net.eval()
traced = torch.jit.trace(net, (torch.zeros(1, 556), torch.zeros(1, 52, dtype=torch.bool)))
traced.save('hearts_ai_match.pt')

# sanity: 556 accepted, 550 rejected
out = traced(torch.rand(2, 556), torch.ones(2, 52, dtype=torch.bool))
try:
    traced(torch.rand(2, 550), torch.ones(2, 52, dtype=torch.bool))
    raise SystemExit("ERROR: match trace accepted 550-dim input")
except RuntimeError:
    pass
print("hearts_ai_match.pt exported (556-dim, rejects 550 as intended)")
