"""Match-to-100 wrapper over the per-deal HeartsEnv (docs/ROADMAP.md phase 1).

A match is a sequence of deals with carried scores, ending when any player
reaches 100+; lowest total wins. The C++ engine stays deal-scoped; this
wrapper owns the match state and APPENDS the 6-dim match context to the
engine's 550-dim observation (see HeartsNetV5.MATCH_CTX_* for the layout).

Determinism/pairing: the engine RNG is consumed only by reset()'s shuffle,
so two MatchEnv instances built from the same seed produce identical deal
SEQUENCES regardless of play - per-match pairing across conditions is valid
(matches may differ in length once scores diverge; deals pair by index).

Seating note: only nets that tolerate >550-dim observations may sit at a
match table - HeartsNetV5 (ignores or consumes the tail) and prefix-slicing
legacy adapters. A bare 550-dim MLP would crash on the appended dims.
"""
import numpy as np

import hearts_env

TARGET = 100
MATCH_CTX_DIM = 6
_MAX_DEALS_NORM = 20.0


def match_ctx_row(match_scores, deals_played, seat):
    """6 dims, rotated to `seat`: [self, left, across, right]/100,
    deals_played/20, leader_distance_to_100/100."""
    rot = np.array([match_scores[(seat + i) % 4] for i in range(4)],
                   dtype=np.float64)
    return np.concatenate([
        rot / float(TARGET),
        [deals_played / _MAX_DEALS_NORM],
        [(TARGET - match_scores.max()) / float(TARGET)],
    ]).astype(np.float32)


def placements_of(match_scores):
    """1 = winner (lowest total); ties share the mean of their ranks."""
    order = np.argsort(match_scores, kind='stable')
    ranks = np.empty(4, dtype=np.float64)
    i = 0
    while i < 4:
        j = i
        while (j + 1 < 4 and
               match_scores[order[j + 1]] == match_scores[order[i]]):
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


class MatchEnv:
    def __init__(self, seed):
        self.env = hearts_env.HeartsEnv(seed=seed)
        self.reset_match()

    def reset_match(self):
        self.match_scores = np.zeros(4, dtype=np.float64)
        self.deals_played = 0
        self.match_over = False
        self.env.reset()

    # --- pass-throughs -----------------------------------------------------
    def get_current_player(self):
        return self.env.get_current_player()

    def get_legal_actions(self):
        return self.env.get_legal_actions()

    def is_passing(self):
        return self.env.is_passing()

    # --- observation with appended match context ---------------------------
    def match_ctx(self, seat):
        return match_ctx_row(self.match_scores, self.deals_played, seat)

    def observe(self):
        obs = np.asarray(self.env.observe(), dtype=np.float32)
        return np.concatenate([obs, self.match_ctx(self.get_current_player())])

    def observe_v2(self):
        """882-dim obs v2 (docs/v6_prereg.md stage 0): the 556-dim v1
        observation as an exact prefix + the 326-dim capture extension
        (position/led/taken-by/seat aggregates). v1 consumers keep using
        observe(); only v6-era recorders/trainers request this."""
        ext = np.asarray(self.env.observe_ext(), dtype=np.float32)
        return np.concatenate([self.observe(), ext])

    def observe_for(self, seat):
        """Any seat's info-honest observation at the current state (what
        that seat could see right now), with its own match context."""
        obs = np.asarray(self.env.observe_for(seat), dtype=np.float32)
        return np.concatenate([obs, self.match_ctx(seat)])

    # --- stepping ----------------------------------------------------------
    def step(self, action):
        """Returns (deal_done, match_done, round_scores_or_None)."""
        if self.match_over:
            raise RuntimeError("step() after match end; call reset_match()")
        result = self.env.step(action)
        if not result.done:
            return False, False, None
        round_scores = np.array(self.env.get_round_scores(), dtype=np.float64)
        self.match_scores += round_scores
        self.deals_played += 1
        if self.match_scores.max() >= TARGET:
            self.match_over = True
            return True, True, round_scores
        self.env.reset()
        return True, False, round_scores

    # --- outcomes ----------------------------------------------------------
    def placements(self):
        assert self.match_over
        return placements_of(self.match_scores)


class MatchVecEnv:
    """Match-state manager over the C++ HeartsVecEnv batch API.

    Mirrors the HeartsVecEnv surface run_cycle_vec-style loops use, with two
    differences: observe_batch appends the per-env match context (obs becomes
    550 + MATCH_CTX_DIM dims), and step_batch additionally reports match
    terminations with placements. The C++ env auto-resets each finished deal;
    this wrapper only resets its own score state at match end, so the next
    deal seamlessly starts the next match.
    """

    def __init__(self, num_envs, seed0):
        self.vec = hearts_env.HeartsVecEnv(num_envs, seed0)
        self.num_envs = num_envs
        self.match_scores = np.zeros((num_envs, 4), dtype=np.float64)
        self.deals_played = np.zeros(num_envs, dtype=np.int64)

    def size(self):
        return self.num_envs

    def current_players(self):
        return self.vec.current_players()

    def legal_mask_batch(self, g):
        return self.vec.legal_mask_batch(g)

    def labels_batch(self, g):
        return self.vec.labels_batch(g)

    def observe_batch(self, g):
        obs = self.vec.observe_batch(g)
        cp = self.vec.current_players()
        ctx = np.stack([
            match_ctx_row(self.match_scores[int(e)],
                          int(self.deals_played[int(e)]), int(cp[int(e)]))
            for e in g])
        return np.concatenate([obs, ctx], axis=1).astype(np.float32)

    def observe_v2_batch(self, g):
        """Obs v2 for v6 learners/pool nets: [classic 550 | match ctx 6 |
        extension 326] = 882 (docs/v6_prereg.md stage 0 layout - exactly
        the training-record obs field)."""
        return np.concatenate(
            [self.observe_batch(g), self.vec.observe_ext_batch(g)],
            axis=1).astype(np.float32)

    def block_events_batch(self, g):
        """Addendum R (docs/exploiter_league_r4_prereg.md §9): (n,2) int32
        [blocker seat, penalty pts] per env since last read, seat -1 = none;
        read-and-clear. Captured by the C++ vec env BEFORE deal auto-reset."""
        return self.vec.block_events_batch(g)

    def step_batch(self, g, actions):
        """Returns (deal_dones, match_dones, placements, round_scores).
        placements[j] is a 4-vector for match-done envs, else None."""
        deal_dones, scores = self.vec.step_batch(g, actions)
        match_dones = np.zeros(len(g), dtype=bool)
        placements = [None] * len(g)
        for j in np.flatnonzero(deal_dones):
            e = int(g[j])
            self.match_scores[e] += np.asarray(scores[j], dtype=np.float64)
            self.deals_played[e] += 1
            if self.match_scores[e].max() >= TARGET:
                match_dones[j] = True
                placements[j] = placements_of(self.match_scores[e])
                self.match_scores[e] = 0.0
                self.deals_played[e] = 0
        return deal_dones, match_dones, placements, scores
