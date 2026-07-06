import torch
import torch.nn as nn
import torch.nn.functional as F

class HeartsNet(nn.Module):
    """
    Actor-Critic Neural Network for the Hearts Reinforcement Learning Environment.
    
    This network uses a shared feature extractor (two 256-neuron hidden layers) 
    and branches into two heads:
      1. Policy Head: Outputs logits for all 52 possible cards in the deck.
      2. Value Head: Outputs a scalar representing the expected penalty points.
    """
    
    def __init__(self):
        super(HeartsNet, self).__init__()
        
        # Shared Feature Extractor
        # Input is the 181-dimensional observation tensor:
        # (52 hand + 52 trick + 52 history + 4 scores + 4 trick_pos + 1 hearts_broken + 16 void_tracker)
        self.shared_fc1 = nn.Linear(181, 256)
        self.shared_fc2 = nn.Linear(256, 256)
        
        # Policy Head (Actor)
        # Outputs 52 logits representing the raw, unnormalized action preferences for the entire deck.
        self.policy_head = nn.Linear(256, 52)
        
        # Value Head (Critic)
        # Outputs a single scalar predicting the total penalty points to be taken from this state.
        self.value_head = nn.Linear(256, 1)

    def forward(self, observation, legal_actions_mask):
        """
        Forward pass of the Actor-Critic network with explicit action masking.
        
        Args:
            observation (torch.Tensor): The 156-dim state tensor. Shape: (batch_size, 156) or (156,)
            legal_actions_mask (torch.Tensor): A boolean mask of shape (batch_size, 52) or (52,)
                                               where True indicates a legal move.
        
        Returns:
            policy_logits (torch.Tensor): Masked logits for the 52 deck actions.
            state_value (torch.Tensor): The expected value (scalar) of the current state.
        """
        
        # 1. Pass through shared hidden layers with ReLU activations
        x = F.relu(self.shared_fc1(observation))
        x = F.relu(self.shared_fc2(x))
        
        # 2. Value Head Calculation
        state_value = self.value_head(x)
        
        # 3. Policy Head Calculation
        logits = self.policy_head(x)
        
        # 4. Action Masking
        # We replace logits of illegal actions with -infinity.
        # When passed through a Softmax function during training/inference, 
        # these -inf values will correctly map to a probability of 0.0.
        masked_logits = logits.masked_fill(~legal_actions_mask, float('-inf'))
        
        return masked_logits, state_value
