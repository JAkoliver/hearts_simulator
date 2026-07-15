import torch
import torch.nn as nn
from hearts_net import net_from_checkpoint

class SearchExport(nn.Module):
    """Trace wrapper exposing all three heads (policy, value, belief) for the
    C++ decision-time search, which needs belief marginals to sample
    determinizations. The plain deployment trace keeps only policy+value.

    A second traced method, `oracle(observation, true_hands)`, exposes the
    oracle value head for truncated-rollout leaf evaluation (the C++ side
    feeds the determinized hands). Kept as a separate method so `forward`
    stays byte-compatible with every existing consumer."""

    def __init__(self, net):
        super(SearchExport, self).__init__()
        self.net = net

    def forward(self, observation, legal_actions_mask):
        return self.net.forward_all(observation, legal_actions_mask)

    def oracle(self, observation, true_hands):
        return self.net.forward_oracle(observation, true_hands)

network = net_from_checkpoint('hearts_model_final.pth')
network.eval()

dummy_obs = torch.zeros(1, 550, dtype=torch.float32)
dummy_mask = torch.zeros(1, 52, dtype=torch.bool)

traced_script_module = torch.jit.trace(network, (dummy_obs, dummy_mask))
traced_script_module.save("hearts_ai_grandmaster.pt")
print("Model successfully exported to hearts_ai_grandmaster.pt")

dummy_hands = torch.zeros(1, 156, dtype=torch.float32)
traced_search = torch.jit.trace_module(
    SearchExport(network),
    {'forward': (dummy_obs, dummy_mask), 'oracle': (dummy_obs, dummy_hands)})
traced_search.save("hearts_ai_search.pt")
print("Search model (policy+value+belief, +oracle method) exported to hearts_ai_search.pt")
