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
        """6 dims, rotated to `seat`: [self, left, across, right]/100,
        deals_played/20, leader_distance_to_100/100."""
        rot = np.array([self.match_scores[(seat + i) % 4] for i in range(4)])
        return np.concatenate([
            rot / float(TARGET),
            [self.deals_played / _MAX_DEALS_NORM],
            [(TARGET - self.match_scores.max()) / float(TARGET)],
        ]).astype(np.float32)

    def observe(self):
        obs = np.asarray(self.env.observe(), dtype=np.float32)
        return np.concatenate([obs, self.match_ctx(self.get_current_player())])

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
        """1 = winner (lowest total); ties share the mean of their ranks."""
        assert self.match_over
        order = np.argsort(self.match_scores, kind='stable')
        ranks = np.empty(4, dtype=np.float64)
        i = 0
        while i < 4:
            j = i
            while (j + 1 < 4 and
                   self.match_scores[order[j + 1]] == self.match_scores[order[i]]):
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return ranks
