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


class V5Block(nn.Module):
    """Pre-LN transformer block with explicit (trace-deterministic) attention."""

    def __init__(self, d_model, num_heads):
        super(V5Block, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        b, t, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(b, t, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b, heads, t, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Fused attention kernel: identical math to the explicit
        # softmax(qk/sqrt(dk))v (default scale IS 1/sqrt(head_dim)), one
        # kernel instead of ~6, and still a plain traced ATen op (the
        # nn.TransformerEncoder fast-path problem does not apply).
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).reshape(b, t, d)
        x = x + self.attn_out(y)
        h = self.norm2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


class HeartsNetV5(nn.Module):
    """Card-token transformer (v5).

    Motivation (2026-07-15): every search-side amplifier — K-scaling, ISMCTS,
    learned leaf evaluators — plateaus at the same ceiling, so the network is
    the binding constraint. The flat-MLP nets see the observation as 550
    anonymous floats; v5 re-encodes the SAME observation as 52 card tokens +
    1 global token and lets attention relate them:

      - each card token = learned card-identity embedding + projection of
        that card's 12 per-card channels sliced from the flat observation
        (in-hand, on-table, played, passed, received, played-by x4, timing)
      - the global token projects the 30 context dims (scores, trick
        position, hearts broken, voids, pass direction, in-passing)
      - policy head reads one logit PER CARD TOKEN (the action space IS the
        card set), belief reads 3 logits per card token, value reads the
        global token.

    Public surface matches HeartsNet exactly (forward / forward_all /
    forward_train over the flat 550-dim observation), so tracing, the C++
    probe, the gate, distillation, and PPO all work unchanged.
    """

    # Card-indexed observation blocks -> per-card channels
    CARD_BLOCKS = [0, 52, 104, 186, 238, 290, 342, 394, 446, 498]  # starts of 52-wide blocks
    N_CARD_CH = len(CARD_BLOCKS)  # 10 sliced channels (identity embedding adds content)
    CTX_START, CTX_END = 156, 186  # scores..in_passing (30 dims)

    def __init__(self, obs_dim=550, d_model=192, num_layers=4, num_heads=6):
        super(HeartsNetV5, self).__init__()
        self.obs_dim = obs_dim
        self.d_model = d_model

        self.card_embed = nn.Embedding(52, d_model)
        self.card_proj = nn.Linear(self.N_CARD_CH, d_model)
        self.ctx_proj = nn.Linear(self.CTX_END - self.CTX_START, d_model)
        # Buffer, not an inline torch.arange: traces bake tensor-creation
        # devices as constants, and a buffer follows .to(device) instead
        self.register_buffer('card_ids', torch.arange(52), persistent=False)

        # Explicit pre-LN attention blocks (not nn.TransformerEncoder, whose
        # runtime fast-path makes torch.jit.trace non-deterministic)
        self.enc_blocks = nn.ModuleList(
            [V5Block(d_model, num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        self.policy_head = nn.Linear(d_model, 1)   # per card token
        self.value_head = nn.Linear(d_model, 1)    # global token
        self.belief_head = nn.Linear(d_model, 3)   # per card token, 3 opponents
        # Interface-compat oracle head (measured uninformative; kept so the
        # distill forward_train code path is architecture-agnostic)
        self.oracle_fc1 = nn.Linear(d_model + 156, d_model)
        self.oracle_fc2 = nn.Linear(d_model, 1)

        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def _tokens(self, observation):
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        b = observation.shape[0]
        # (batch, 52, n_channels): stack the card-indexed blocks
        chans = torch.stack([observation[:, s:s + 52] for s in self.CARD_BLOCKS], dim=2)
        cards = self.card_embed(self.card_ids).unsqueeze(0).expand(b, 52, self.d_model) \
            + self.card_proj(chans)
        ctx = self.ctx_proj(observation[:, self.CTX_START:self.CTX_END]).unsqueeze(1)
        x = torch.cat([ctx, cards], dim=1)  # (batch, 53, d)
        for block in self.enc_blocks:
            x = block(x)
        return self.final_norm(x)

    def _heads(self, x, legal_actions_mask):
        if legal_actions_mask.dim() == 1:
            legal_actions_mask = legal_actions_mask.unsqueeze(0)
        global_tok = x[:, 0, :]
        card_toks = x[:, 1:, :]  # (batch, 52, d)
        logits = self.policy_head(card_toks).squeeze(-1)  # (batch, 52)
        masked_logits = logits.masked_fill(~legal_actions_mask, float('-inf'))
        state_value = self.value_head(global_tok)
        # belief: per-card 3 outputs -> label layout (3 opponents x 52 cards)
        bel = self.belief_head(card_toks)                    # (batch, 52, 3)
        belief_logits = bel.transpose(1, 2).reshape(-1, 156)  # (batch, 3*52)
        return masked_logits, state_value, belief_logits, global_tok

    def forward(self, observation, legal_actions_mask):
        x = self._tokens(observation)
        masked_logits, state_value, _, _ = self._heads(x, legal_actions_mask)
        return masked_logits, state_value

    def forward_all(self, observation, legal_actions_mask):
        x = self._tokens(observation)
        masked_logits, state_value, belief_logits, _ = self._heads(x, legal_actions_mask)
        return masked_logits, state_value, belief_logits

    def forward_train(self, observation, legal_actions_mask, true_hands):
        x = self._tokens(observation)
        masked_logits, state_value, belief_logits, g = self._heads(x, legal_actions_mask)
        h = F.gelu(self.oracle_fc1(torch.cat([g, true_hands], dim=-1)))
        return masked_logits, state_value, belief_logits, self.oracle_fc2(h)

    def forward_oracle(self, observation, true_hands):
        x = self._tokens(observation)
        h = F.gelu(self.oracle_fc1(torch.cat([x[:, 0, :], true_hands], dim=-1)))
        return self.oracle_fc2(h)


def net_from_checkpoint(path, map_location=None):
    """Construct the right network class at the right size for a checkpoint.

    Dispatches on state-dict keys: 'card_embed.weight' -> HeartsNetV5
    (d_model from the embedding, layer count from encoder keys);
    'input_fc.weight' -> HeartsNet MLP (width/blocks as before). Tolerates
    checkpoints saved before the oracle head existed.
    """
    sd = torch.load(path, weights_only=True, map_location=map_location)
    if 'card_embed.weight' in sd:
        d_model = sd['card_embed.weight'].shape[1]
        num_layers = 1 + max(int(k.split('.')[1]) for k in sd
                             if k.startswith('enc_blocks.'))
        heads = max(1, d_model // 32)
        net = HeartsNetV5(d_model=d_model, num_layers=num_layers, num_heads=heads)
    else:
        width, obs_dim = sd['input_fc.weight'].shape
        num_blocks = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.'))
        net = HeartsNet(obs_dim=obs_dim, width=width, num_blocks=num_blocks)
    net.load_state_dict(sd, strict=False)
    return net
