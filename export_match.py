"""Export the 556-dim MATCH trace of the current baseline.

Traced with a 556-dim dummy so the match_proj branch is baked in: this
trace REQUIRES 556-dim input (engine obs + 6 match-ctx dims) and is what
SearchEval's --match-model mode consumes. Not produced by export.py -
regenerate after promotions when running match-bridge measurements.
"""
import torch
import torch.nn as nn

from hearts_net import net_from_checkpoint
from train_equity import EquityNet


class SearchExport(nn.Module):
    """Same trace surface as export.py's SearchExport (not imported from
    there: export.py runs its export at module level)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, observation, legal_actions_mask):
        return self.net.forward_all(observation, legal_actions_mask)

    def oracle(self, observation, true_hands):
        return self.net.forward_oracle(observation, true_hands)

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
print("hearts_ai_match.pt exported (556-dim raw, rejects 550 as intended)")

# 556-dim SEARCH trace (policy+value+belief, +oracle) for match-aware search:
# determinization sampling needs the belief head, and in match-aware mode
# every net call carries the acting seat's context.
traced_search = torch.jit.trace_module(
    SearchExport(net),
    {'forward': (torch.zeros(1, 556), torch.zeros(1, 52, dtype=torch.bool)),
     'oracle': (torch.zeros(1, 556), torch.zeros(1, 156))})
traced_search.save('hearts_ai_search_match.pt')
print("hearts_ai_search_match.pt exported (556-dim search trace w/ belief)")

# Equity net -> TorchScript for C++ leaf scoring (logits out; softmax C++-side)
ck = torch.load('equity_v1.pth', weights_only=True)
eq = EquityNet(ck['in_dim'])
eq.load_state_dict(ck['state_dict'])
eq.eval()
torch.jit.trace(eq, torch.zeros(1, 10)).save('hearts_equity.pt')
print("hearts_equity.pt exported (10 -> 4 logits)")
