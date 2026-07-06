import torch
from hearts_net import HeartsNet

network = HeartsNet()
network.load_state_dict(torch.load('hearts_model_final.pth', weights_only=True))
network.eval()

dummy_obs = torch.zeros(1, 181, dtype=torch.float32)
dummy_mask = torch.zeros(1, 52, dtype=torch.bool)

traced_script_module = torch.jit.trace(network, (dummy_obs, dummy_mask))
traced_script_module.save("hearts_ai_grandmaster.pt")
print("Model successfully exported to hearts_ai_grandmaster.pt")
