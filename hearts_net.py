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

    Architecture (v4): a LayerNorm residual trunk with three heads.
      - Input projection: 550 -> width
      - num_blocks pre-LN residual blocks (width -> width)
      - Final LayerNorm, then:
          1. Policy Head: logits for all 52 possible cards in the deck.
          2. Value Head: scalar estimating the expected RELATIVE round reward
             (table average minus own score), so positive = better than the table.
          3. Auxiliary Belief Head (training only): 3x52 logits predicting which
             cards each RELATIVE opponent (left/across/right) currently holds.
             Trained supervised against ground truth from self-play; forces the
             shared trunk to learn hidden-hand inference. Not used at play time
             and excluded from the traced deployment graph.

    Initialization: orthogonal weights throughout (gain sqrt(2)); each residual
    branch's last layer is zero-initialized so every block starts as identity;
    the policy head uses gain 0.01 so the initial policy is near-uniform over
    legal actions, which keeps early PPO updates well-conditioned.

    NOTE: v2 checkpoints are NOT weight-compatible with v1 (the 181->256->256
    MLP). v1 .pth files cannot be loaded into this class. The traced TorchScript
    deployment asset (hearts_ai_grandmaster.pt) is self-contained and can still
    be loaded with torch.jit.load() regardless of architecture.
    """

    def __init__(self, obs_dim=550, width=512, num_blocks=3):
        super(HeartsNet, self).__init__()

        # Input is the 550-dimensional observation tensor:
        # (52 hand + 52 trick + 52 history + 4 scores + 4 trick_pos + 1 hearts_broken
        #  + 16 void_tracker + 4 pass_direction + 1 in_passing + 52 cards_i_passed
        #  + 52 cards_i_received + 4x52 who_played_what(rel seats) + 52 play_timing)
        # The same policy head serves both phases: during passing, the chosen
        # "action" is the card to pass (legality mask restricts to the hand).
        self.input_fc = nn.Linear(obs_dim, width)
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(width)

        # Policy Head (Actor)
        self.policy_head = nn.Linear(width, 52)

        # Value Head (Critic)
        self.value_head = nn.Linear(width, 1)

        # Auxiliary Belief Head: per relative opponent, per card, "do they hold it?"
        self.belief_head = nn.Linear(width, 156)

        # Oracle Value Head: predicts the final relative round reward GIVEN the
        # true opponent hands (156 planes, same layout as the belief labels).
        # The hands enter only this branch - never the trunk - so no hidden
        # information can leak into the policy/value/belief heads. Purpose:
        # leaf evaluator for determinized search, where the sampled hands ARE
        # known and a visible-info value provably cannot distinguish
        # determinizations (measured 2026-07-14: truncated search collapsed
        # +6 pts/deal with a visible-info evaluator).
        self.oracle_fc1 = nn.Linear(width + 156, width)
        self.oracle_fc2 = nn.Linear(width, 1)

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
        nn.init.orthogonal_(self.belief_head.weight, gain=1.0)
        nn.init.orthogonal_(self.oracle_fc2.weight, gain=1.0)

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
        x = self._trunk(observation)

        state_value = self.value_head(x)
        logits = self.policy_head(x)

        # Replace logits of illegal actions with -infinity so Softmax maps
        # them to a probability of exactly 0.0.
        masked_logits = logits.masked_fill(~legal_actions_mask, float('-inf'))

        return masked_logits, state_value

    def _trunk(self, observation):
        x = F.gelu(self.input_fc(observation))
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward_all(self, observation, legal_actions_mask):
        """Training-time forward: one trunk pass feeding all three heads.

        Returns (masked_policy_logits, state_value, belief_logits) where
        belief_logits has shape (batch, 156) = 3 relative opponents x 52 cards.
        """
        x = self._trunk(observation)
        state_value = self.value_head(x)
        masked_logits = self.policy_head(x).masked_fill(~legal_actions_mask, float('-inf'))
        belief_logits = self.belief_head(x)
        return masked_logits, state_value, belief_logits

    def forward_train(self, observation, legal_actions_mask, true_hands):
        """Distillation forward: one trunk pass feeding all four heads.

        true_hands: (batch, 156) float 0/1 planes of the opponents' actual
        hands (same layout as the belief labels). Returns
        (masked_policy_logits, state_value, belief_logits, oracle_value).
        """
        x = self._trunk(observation)
        state_value = self.value_head(x)
        masked_logits = self.policy_head(x).masked_fill(~legal_actions_mask, float('-inf'))
        belief_logits = self.belief_head(x)
        h = F.gelu(self.oracle_fc1(torch.cat([x, true_hands], dim=-1)))
        oracle_value = self.oracle_fc2(h)
        return masked_logits, state_value, belief_logits, oracle_value

    def forward_oracle(self, observation, true_hands):
        """Oracle value only: expected final relative round reward given the
        true (or determinized) opponent hands."""
        x = self._trunk(observation)
        h = F.gelu(self.oracle_fc1(torch.cat([x, true_hands], dim=-1)))
        return self.oracle_fc2(h)


def net_from_checkpoint(path, map_location=None):
    """Construct a HeartsNet matching a checkpoint's actual dimensions.

    Infers obs_dim/width/num_blocks from the state dict (checkpoints of any
    width/depth load without knowing their config), tolerating checkpoints
    saved before the oracle head existed.
    """
    sd = torch.load(path, weights_only=True, map_location=map_location)
    width, obs_dim = sd['input_fc.weight'].shape
    num_blocks = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.'))
    net = HeartsNet(obs_dim=obs_dim, width=width, num_blocks=num_blocks)
    net.load_state_dict(sd, strict=False)
    return net
