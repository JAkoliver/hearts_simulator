import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Pre-LayerNorm residual MLP block: x + Linear(GELU(Linear(LN(x))))."""

    def __init__(self, width):
        super(ResidualBlock, self).__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x):
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + h

class HeartsNet(nn.Module):
    """
    Actor-Critic Neural Network for the Hearts Reinforcement Learning Environment.

    Architecture (v2): a LayerNorm residual trunk replacing the v1 two-layer MLP.
      - Input projection: 238 -> width
      - num_blocks pre-LN residual blocks (width -> width)
      - Final LayerNorm, then two heads:
          1. Policy Head: logits for all 52 possible cards in the deck.
          2. Value Head: scalar estimating the expected RELATIVE round reward
             (table average minus own score), so positive = better than the table.

    Initialization: orthogonal weights throughout (gain sqrt(2)); each residual
    branch's last layer is zero-initialized so every block starts as identity;
    the policy head uses gain 0.01 so the initial policy is near-uniform over
    legal actions, which keeps early PPO updates well-conditioned.

    NOTE: v2 checkpoints are NOT weight-compatible with v1 (the 181->256->256
    MLP). v1 .pth files cannot be loaded into this class. The traced TorchScript
    deployment asset (hearts_ai_grandmaster.pt) is self-contained and can still
    be loaded with torch.jit.load() regardless of architecture.
    """

    def __init__(self, obs_dim=238, width=512, num_blocks=3):
        super(HeartsNet, self).__init__()

        # Input is the 238-dimensional observation tensor:
        # (52 hand + 52 trick + 52 history + 4 scores + 4 trick_pos + 1 hearts_broken
        #  + 16 void_tracker + 4 pass_direction + 1 in_passing + 52 cards_i_passed)
        # The same policy head serves both phases: during passing, the chosen
        # "action" is the card to pass (legality mask restricts to the hand).
        self.input_fc = nn.Linear(obs_dim, width)
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(width)

        # Policy Head (Actor)
        self.policy_head = nn.Linear(width, 52)

        # Value Head (Critic)
        self.value_head = nn.Linear(width, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        # Zero the residual branches so each block starts as the identity map
        for block in self.blocks:
            nn.init.zeros_(block.fc2.weight)
        # Near-uniform initial policy, unit-scale initial value estimates
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, observation, legal_actions_mask):
        """
        Forward pass of the Actor-Critic network with explicit action masking.

        Args:
            observation (torch.Tensor): The 238-dim state tensor. Shape: (batch_size, 238) or (238,)
            legal_actions_mask (torch.Tensor): A boolean mask of shape (batch_size, 52) or (52,)
                                               where True indicates a legal move.

        Returns:
            policy_logits (torch.Tensor): Masked logits for the 52 deck actions.
            state_value (torch.Tensor): The expected value (scalar) of the current state.
        """
        x = F.gelu(self.input_fc(observation))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        state_value = self.value_head(x)
        logits = self.policy_head(x)

        # Replace logits of illegal actions with -infinity so Softmax maps
        # them to a probability of exactly 0.0.
        masked_logits = logits.masked_fill(~legal_actions_mask, float('-inf'))

        return masked_logits, state_value
