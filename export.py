import torch
import torch.nn as nn
from hearts_net import HeartsNet

class SearchExport(nn.Module):
    """Trace wrapper exposing all three heads (policy, value, belief) for the
    C++ decision-time search, which needs belief marginals to sample
    determinizations. The plain deployment trace keeps only policy+value."""

    def __init__(self, net):
        super(SearchExport, self).__init__()
        self.net = net

    def forward(self, observation, legal_actions_mask):
        return self.net.forward_all(observation, legal_actions_mask)

network = HeartsNet()
network.load_state_dict(torch.load('hearts_model_final.pth', weights_only=True))
network.eval()

dummy_obs = torch.zeros(1, 550, dtype=torch.float32)
dummy_mask = torch.zeros(1, 52, dtype=torch.bool)

traced_script_module = torch.jit.trace(network, (dummy_obs, dummy_mask))
traced_script_module.save("hearts_ai_grandmaster.pt")
print("Model successfully exported to hearts_ai_grandmaster.pt")

traced_search = torch.jit.trace(SearchExport(network), (dummy_obs, dummy_mask))
traced_search.save("hearts_ai_search.pt")
print("Search model (policy+value+belief) exported to hearts_ai_search.pt")
