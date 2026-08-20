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

    # Match-to-100 context, APPENDED past the 550-dim per-deal observation
    # (docs/ROADMAP.md phase 1). Layout (all /100 or /20 normalized, rotated
    # to the observing seat): [self, left, across, right match scores,
    # deals_elapsed, distance_of_leader_to_100]. The projection is
    # ZERO-INITIALIZED so a net extended from a per-deal checkpoint is
    # behavior-identical until training moves these weights; per-deal callers
    # passing plain 550-dim observations skip the term entirely.
    MATCH_CTX_START = 550
    MATCH_CTX_DIM = 6

    def __init__(self, obs_dim=550, d_model=192, num_layers=4, num_heads=6):
        super(HeartsNetV5, self).__init__()
        self.obs_dim = obs_dim
        self.d_model = d_model

        self.card_embed = nn.Embedding(52, d_model)
        self.card_proj = nn.Linear(self.N_CARD_CH, d_model)
        self.ctx_proj = nn.Linear(self.CTX_END - self.CTX_START, d_model)
        self.match_proj = nn.Linear(self.MATCH_CTX_DIM, d_model)
        nn.init.zeros_(self.match_proj.weight)
        nn.init.zeros_(self.match_proj.bias)
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
        ctx = self.ctx_proj(observation[:, self.CTX_START:self.CTX_END])
        if observation.shape[-1] >= self.MATCH_CTX_START + self.MATCH_CTX_DIM:
            ctx = ctx + self.match_proj(
                observation[:, self.MATCH_CTX_START:
                            self.MATCH_CTX_START + self.MATCH_CTX_DIM])
        x = torch.cat([ctx.unsqueeze(1), cards], dim=1)  # (batch, 53, d)
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


class HeartsNetV6(nn.Module):
    """Card+seat-token transformer (v6, docs/v6_prereg.md stage 1).

    Extends v5's token scheme on the 882-dim obs v2: 1 global + 4 SEAT
    tokens + 52 card tokens. Motivation (measured): moon threat, trick
    control and match targeting are SEAT properties, and v5 had no seat
    entities - worse, the v1 ctx score/void blocks are in ABSOLUTE seats
    while nothing tells the net its own seat index, so per-opponent deal
    points were structurally unattributable. Obs v2's relative-frame
    capture planes fix the information; the seat tokens give it a place
    to live.

      - card tokens: v5's 10 channels + 6 obs-v2 channels (within-trick
        position, led flag, taken-by x4 relative planes)
      - seat tokens (relative seats 0=self,1=left,2=across,3=right):
        identity embedding + [match score, tricks won, moon-alive,
        deal points taken (derived: penalty-weighted taken-by plane),
        leads-current-trick (derived)]
      - global token: v5's 30 ctx dims + 6 match ctx + obs-v2 tail
        (hearts-unseen, QS one-hot)
      - heads: policy per card token, value on global, belief per card
        token (all as v5); NEW training-only aux heads on seat tokens -
        moon head (did this seat moon the deal) and per-seat final deal
        points. The v5 oracle head is DELETED (measured uninformative).

    Requires the full 882-dim observation; shorter inputs raise (no
    silent zero-padding - a v1 pipeline feeding v6 is a bug).
    """

    CARD_BLOCKS = HeartsNetV5.CARD_BLOCKS + [556, 608, 660, 712, 764, 816]
    N_CARD_CH = len(CARD_BLOCKS)                      # 16
    CTX_START, CTX_END = 156, 186
    MATCH_CTX_START, MATCH_CTX_DIM = 550, 6
    EXT_TRICKS_WON = 868      # 4: tricks won per relative seat
    EXT_MOON_ALIVE = 872      # 4: moon-alive per relative seat
    EXT_TAIL_START, EXT_TAIL_END = 876, 882   # hearts-unseen + QS one-hot
    TAKEN_BY_START = 660      # 4x52 relative taken-by planes
    OBS_DIM = 882
    N_SEAT_FEATS = 5

    def __init__(self, obs_dim=882, d_model=448, num_layers=8, num_heads=8):
        super(HeartsNetV6, self).__init__()
        self.obs_dim = obs_dim
        self.d_model = d_model

        self.card_embed = nn.Embedding(52, d_model)
        self.card_proj = nn.Linear(self.N_CARD_CH, d_model)
        self.seat_embed = nn.Embedding(4, d_model)
        self.seat_proj = nn.Linear(self.N_SEAT_FEATS, d_model)
        self.ctx_proj = nn.Linear(self.CTX_END - self.CTX_START, d_model)
        self.match_proj = nn.Linear(self.MATCH_CTX_DIM, d_model)
        self.tail_proj = nn.Linear(self.EXT_TAIL_END - self.EXT_TAIL_START,
                                   d_model)
        self.register_buffer('card_ids', torch.arange(52), persistent=False)
        self.register_buffer('seat_ids', torch.arange(4), persistent=False)
        # Penalty values per card (/26) for the derived points feature
        pen = torch.zeros(52)
        pen[39:52] = 1.0 / 26.0
        pen[36] = 13.0 / 26.0
        self.register_buffer('penalty', pen, persistent=False)

        self.enc_blocks = nn.ModuleList(
            [V5Block(d_model, num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        self.policy_head = nn.Linear(d_model, 1)   # per card token
        self.value_head = nn.Linear(d_model, 1)    # global token
        self.belief_head = nn.Linear(d_model, 3)   # per card token
        # Aux heads (training-only), one output per SEAT token
        self.moon_head = nn.Linear(d_model, 1)
        self.points_head = nn.Linear(d_model, 1)

        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def _tokens(self, observation):
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        if observation.shape[-1] != self.OBS_DIM:
            raise ValueError(
                f'HeartsNetV6 requires the {self.OBS_DIM}-dim obs v2, '
                f'got {observation.shape[-1]}')
        b = observation.shape[0]
        chans = torch.stack(
            [observation[:, s:s + 52] for s in self.CARD_BLOCKS], dim=2)
        cards = self.card_embed(self.card_ids).unsqueeze(0) \
            .expand(b, 52, self.d_model) + self.card_proj(chans)
        # seat features, relative frame throughout
        taken = observation[:, self.TAKEN_BY_START:
                            self.TAKEN_BY_START + 208].reshape(b, 4, 52)
        pts = taken @ self.penalty                          # (b, 4)
        # leads-current-trick: the led card of the trick in progress is
        # (led flag AND on the table now); dot with each seat's
        # who-played plane attributes it
        led_now = observation[:, 608:660] * observation[:, 52:104]  # (b,52)
        who = observation[:, 290:498].reshape(b, 4, 52)
        leads = (who * led_now.unsqueeze(1)).sum(-1)        # (b, 4)
        feats = torch.stack([
            observation[:, self.MATCH_CTX_START:self.MATCH_CTX_START + 4],
            observation[:, self.EXT_TRICKS_WON:self.EXT_TRICKS_WON + 4],
            observation[:, self.EXT_MOON_ALIVE:self.EXT_MOON_ALIVE + 4],
            pts, leads], dim=2)                             # (b, 4, 5)
        seats = self.seat_embed(self.seat_ids).unsqueeze(0) \
            .expand(b, 4, self.d_model) + self.seat_proj(feats)
        ctx = self.ctx_proj(observation[:, self.CTX_START:self.CTX_END]) \
            + self.match_proj(observation[:, self.MATCH_CTX_START:
                                          self.MATCH_CTX_START
                                          + self.MATCH_CTX_DIM]) \
            + self.tail_proj(observation[:, self.EXT_TAIL_START:
                                         self.EXT_TAIL_END])
        x = torch.cat([ctx.unsqueeze(1), seats, cards], dim=1)  # (b, 57, d)
        for block in self.enc_blocks:
            x = block(x)
        return self.final_norm(x)

    def _heads(self, x, legal_actions_mask):
        if legal_actions_mask.dim() == 1:
            legal_actions_mask = legal_actions_mask.unsqueeze(0)
        global_tok = x[:, 0, :]
        card_toks = x[:, 5:, :]                       # (batch, 52, d)
        logits = self.policy_head(card_toks).squeeze(-1)
        masked_logits = logits.masked_fill(~legal_actions_mask, float('-inf'))
        state_value = self.value_head(global_tok)
        bel = self.belief_head(card_toks)             # (batch, 52, 3)
        belief_logits = bel.transpose(1, 2).reshape(-1, 156)
        return masked_logits, state_value, belief_logits

    def forward(self, observation, legal_actions_mask):
        x = self._tokens(observation)
        masked_logits, state_value, _ = self._heads(x, legal_actions_mask)
        return masked_logits, state_value

    def forward_all(self, observation, legal_actions_mask):
        x = self._tokens(observation)
        return self._heads(x, legal_actions_mask)

    def forward_aux(self, observation, legal_actions_mask):
        """Training forward: standard heads + the seat-token aux heads.

        Returns (masked_logits, value, belief_logits, moon_logits[b,4],
        seat_points[b,4]) - moon labels are per RELATIVE seat 'this seat
        shot the moon this deal'; points labels are final round_scores
        (/26) in the relative frame."""
        x = self._tokens(observation)
        masked_logits, state_value, belief_logits = \
            self._heads(x, legal_actions_mask)
        seat_toks = x[:, 1:5, :]
        moon_logits = self.moon_head(seat_toks).squeeze(-1)
        seat_points = self.points_head(seat_toks).squeeze(-1)
        return masked_logits, state_value, belief_logits, \
            moon_logits, seat_points


class HeartsNetV5Ext(HeartsNetV5):
    """League round 5 (docs/exploiter_league_r5_prereg.md §3.3): the v5
    champion plus ZERO-INITIALIZED adapters over the obs-v2 extension.

    Card tokens gain 6 extra channels (within-trick position, led flag,
    taken-by x4 RELATIVE planes) through `ext_card_proj`; the global token
    gains 14 dims (tricks won x4, moon-alive x4, hearts unseen, Q-spade
    status one-hot x5) through `ext_ctx_proj`. Both start at exactly zero,
    so a champion checkpoint loaded into this class is BIT-IDENTICAL to
    HeartsNetV5 on the 556 prefix until training moves the adapters (the
    verified match_proj / extended-v5 pattern). obs_dim = 882; a 556-dim
    caller is a bug (raises), same contract as HeartsNetV6.
    """
    EXT_CARD_BLOCKS = [556, 608, 660, 712, 764, 816]   # 6 x 52
    N_EXT_CARD_CH = len(EXT_CARD_BLOCKS)
    EXT_TRICKS_WON = 868      # 4
    EXT_MOON_ALIVE = 872      # 4
    EXT_TAIL_START, EXT_TAIL_END = 876, 882   # hearts-unseen + QS one-hot (6)
    N_EXT_CTX = 4 + 4 + 6      # 14
    OBS_DIM = 882

    def __init__(self, obs_dim=882, d_model=320, num_layers=6, num_heads=10):
        super().__init__(obs_dim=556, d_model=d_model, num_layers=num_layers,
                         num_heads=num_heads)
        self.obs_dim = 882
        self.ext_card_proj = nn.Linear(self.N_EXT_CARD_CH, d_model)
        self.ext_ctx_proj = nn.Linear(self.N_EXT_CTX, d_model)
        for m in (self.ext_card_proj, self.ext_ctx_proj):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _tokens(self, observation):
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        if observation.shape[-1] != self.OBS_DIM:
            raise ValueError(
                f'HeartsNetV5Ext requires the {self.OBS_DIM}-dim obs v2, '
                f'got {observation.shape[-1]}')
        b = observation.shape[0]
        chans = torch.stack([observation[:, s:s + 52] for s in self.CARD_BLOCKS], dim=2)
        ext = torch.stack([observation[:, s:s + 52] for s in self.EXT_CARD_BLOCKS], dim=2)
        cards = self.card_embed(self.card_ids).unsqueeze(0).expand(b, 52, self.d_model)             + self.card_proj(chans) + self.ext_card_proj(ext)
        ctx = self.ctx_proj(observation[:, self.CTX_START:self.CTX_END])             + self.match_proj(observation[:, self.MATCH_CTX_START:
                                          self.MATCH_CTX_START + self.MATCH_CTX_DIM])             + self.ext_ctx_proj(torch.cat([
                observation[:, self.EXT_TRICKS_WON:self.EXT_TRICKS_WON + 4],
                observation[:, self.EXT_MOON_ALIVE:self.EXT_MOON_ALIVE + 4],
                observation[:, self.EXT_TAIL_START:self.EXT_TAIL_END]], dim=1))
        x = torch.cat([ctx.unsqueeze(1), cards], dim=1)  # (batch, 53, d)
        for block in self.enc_blocks:
            x = block(x)
        return self.final_norm(x)


class HeartsHybrid(nn.Module):
    """Two raw nets that SWITCH by state (docs/hybrid_specialist_probe.md):
    the CHAMPION (556 inputs) plays every decision except moon-alive THREAT
    states, where the SPECIALIST (an obs-v2 net, 882 inputs) plays instead.
    A hard gate, not a perturbation - neither net's weights change.

    Gate 'threat' (primary): some OPPONENT relative seat is moon-alive
    (obs-v2 dims 872-875, seats 1-3) AND that seat has taken >= 1 penalty
    point (from the taken-by planes 660-867 x penalty). Passing-phase and
    no-points states never fire (everyone is trivially alive there).
    Gate 'any_alive' (secondary): any opponent moon-alive flag set.
    obs_dim = 882 (needs obs v2); the champion sees the 556 prefix.
    """
    OBS_DIM = 882
    _PEN = None

    def __init__(self, champion, specialist, gate='threat', router=None):
        super().__init__()
        self.champion = champion
        self.specialist = specialist
        # optional separate ROUTER net for 'moonhead' gates (decouples the
        # detector from the specialist: e.g. arm b's moon head routing r1-t3);
        # default = the specialist's own aux head
        self.router = router
        self.gate = gate
        self.obs_dim = 882
        pen = torch.zeros(52); pen[39:52] = 1.0; pen[36] = 13.0
        self.register_buffer('pen', pen, persistent=False)

    # Gate grammar (docs/hybrid_specialist_probe.md ladder):
    #   'threat'        opponent moon-alive AND >= 1 pt (the probe's primary)
    #   'threat:N'      opponent moon-alive AND >= N pts held by that seat
    #   'any_alive'     any opponent moon-alive flag
    #   'moonhead:T'    specialist's own aux moon head: max_opp sigmoid(logit) > T
    #   'uncertain:D'   threat (>=1 pt) AND champion top-2 logit gap < D
    def gate_mask(self, observation, lc=None):
        alive = observation[:, 872:876] > 0.5                       # rel seats 0..3
        kind, _, arg = self.gate.partition(':')
        if kind == 'any_alive':
            return alive[:, 1:].any(dim=1)
        taken = observation[:, 660:868].reshape(-1, 4, 52)
        pts = taken @ self.pen                                       # (b, 4)
        if kind == 'threat':
            thr = float(arg) if arg else 1.0
            return (alive[:, 1:] & (pts[:, 1:] >= thr)).any(dim=1)
        if kind == 'moonhead':
            tau = float(arg)
            det = self.router if self.router is not None else self.specialist
            _, _, _, moon_logits, _ = det.forward_aux(
                observation, torch.ones(observation.shape[0], 52, dtype=torch.bool,
                                        device=observation.device))
            p = torch.sigmoid(moon_logits[:, 1:])                    # opponents
            return (p.max(dim=1).values > tau) & alive[:, 1:].any(dim=1)
        if kind == 'uncertain':
            d = float(arg)
            threat = (alive[:, 1:] & (pts[:, 1:] >= 1.0)).any(dim=1)
            top2 = lc.masked_fill(~torch.isfinite(lc), -1e9).topk(2, dim=1).values
            gap = top2[:, 0] - top2[:, 1]
            return threat & (gap < d)
        raise ValueError(f'unknown gate {self.gate}')

    def forward(self, observation, legal_actions_mask):
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        if legal_actions_mask.dim() == 1:
            legal_actions_mask = legal_actions_mask.unsqueeze(0)
        if observation.shape[-1] != self.OBS_DIM:
            raise ValueError(f'HeartsHybrid requires 882-dim obs v2, got {observation.shape[-1]}')
        lc, vc = self.champion(observation[:, :556], legal_actions_mask)
        g = self.gate_mask(observation, lc)
        sdim = 882 if getattr(self.specialist, 'obs_dim', 556) == 882 else 556
        ls, vs = self.specialist(observation[:, :sdim], legal_actions_mask)
        gm = g.unsqueeze(1)
        return torch.where(gm, ls, lc), torch.where(gm, vs, vc)

    def forward_all(self, observation, legal_actions_mask):
        logits, value = self.forward(observation, legal_actions_mask)
        _, _, bel = self.champion.forward_all(observation[:, :556], legal_actions_mask)
        return logits, value, bel


def save_hybrid(path, champion_ckpt, specialist_ckpt, gate='threat', router_ckpt=None):
    d = {'hybrid': True, 'gate': gate,
         'champion_sd': torch.load(champion_ckpt, weights_only=True, map_location='cpu'),
         'specialist_sd': torch.load(specialist_ckpt, weights_only=True, map_location='cpu')}
    if router_ckpt:
        d['router_sd'] = torch.load(router_ckpt, weights_only=True, map_location='cpu')
    torch.save(d, path)


def net_from_checkpoint(path, map_location=None):
    """Construct the right network class at the right size for a checkpoint.

    Dispatches on state-dict keys: 'card_embed.weight' -> HeartsNetV5
    (d_model from the embedding, layer count from encoder keys);
    'input_fc.weight' -> HeartsNet MLP (width/blocks as before). Tolerates
    checkpoints saved before the oracle head existed.
    """
    sd = torch.load(path, weights_only=True, map_location=map_location)
    if isinstance(sd, dict) and sd.get('hybrid'):
        # in-memory round trip (no temp files: 12 pool workers x 2 files per
        # hybrid was an avoidable startup I/O burst - 2026-08-18 incident)
        import io
        parts = []
        for k in ('champion_sd', 'specialist_sd', 'router_sd'):
            if k not in sd:
                parts.append(None); continue
            buf = io.BytesIO(); torch.save(sd[k], buf); buf.seek(0)
            parts.append(net_from_checkpoint(buf, map_location))
        net = HeartsHybrid(parts[0], parts[1], gate=sd.get('gate', 'threat'), router=parts[2])
        net.eval()
        return net
    if 'ext_card_proj.weight' in sd:          # league r5 adapter net (882)
        d_model = sd['card_embed.weight'].shape[1]
        num_layers = 1 + max(int(k.split('.')[1]) for k in sd
                             if k.startswith('enc_blocks.'))
        heads = max(1, d_model // 32)
        net = HeartsNetV5Ext(d_model=d_model, num_layers=num_layers,
                             num_heads=heads)
    elif 'seat_embed.weight' in sd:
        d_model = sd['card_embed.weight'].shape[1]
        num_layers = 1 + max(int(k.split('.')[1]) for k in sd
                             if k.startswith('enc_blocks.'))
        heads = max(1, d_model // 56)
        net = HeartsNetV6(d_model=d_model, num_layers=num_layers,
                          num_heads=heads)
    elif 'card_embed.weight' in sd:
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
