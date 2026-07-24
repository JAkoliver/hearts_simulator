"""Web server for human-vs-AI match Hearts (docs/ROADMAP.md: human calibration).

One human seat vs three AI seats (the current promoted baseline, raw
policy, clearly labeled in the UI) playing matches to 100 on the exact
match rules the training uses (hearts_match_env.MatchEnv).

MVP surface:
  POST /api/new                 -> {sid, state}
  GET  /api/state/{sid}         -> state for the human seat
  POST /api/play/{sid} {card}   -> apply human action, run AI turns, -> state

State is built from server-side bookkeeping (trick history, scores) plus
the engine's hand/legal-move view for the human seat. Every match appends
one JSON line of full telemetry to match_logs.jsonl.

Run:  python -m uvicorn hearts_web.server:app --host 0.0.0.0 --port 8642
"""
import json
import os
import secrets
import sys
import threading
import time

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hearts_match_env import MatchEnv, TARGET  # noqa: E402
from hearts_net import net_from_checkpoint  # noqa: E402

SUITS = 'CDSH'
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'match_logs.jsonl')

app = FastAPI(title="Hearts vs AI")

_net = net_from_checkpoint('hearts_model_final.pth')
_net.eval()
_net_lock = threading.Lock()
_sessions = {}
_sessions_lock = threading.Lock()


def card_name(idx):
    return RANKS[idx % 13] + SUITS[idx // 13]


class Session:
    def __init__(self):
        self.sid = secrets.token_urlsafe(12)
        self.menv = MatchEnv(seed=secrets.randbits(31))
        self.human_seat = secrets.randbelow(4)
        self.trick = []           # [(seat, card)] of the current trick
        self.deal_no = 1
        self.passed_cards = []    # human's picks this pass phase
        self.log = {'start': time.time(), 'human_seat': self.human_seat,
                    'deals': [], 'actions': 0}
        self.finished = False
        self.lock = threading.Lock()

    # -- engine helpers -----------------------------------------------------
    def _legal(self):
        return [a for a in self.menv.get_legal_actions() if a != -1]

    def _ai_action(self):
        obs = torch.from_numpy(self.menv.observe()).unsqueeze(0)
        mask = torch.zeros((1, 52), dtype=torch.bool)
        for a in self._legal():
            mask[0, a] = True
        with _net_lock, torch.no_grad():
            logits, _ = _net(obs, mask)
        return int(torch.argmax(logits, dim=1).item())

    def _apply(self, seat, action):
        if not self.menv.is_passing():
            self.trick.append((seat, action))
            if len(self.trick) > 4:
                self.trick = self.trick[-4:]
        deal_done, match_done, round_scores = self.menv.step(action)
        self.log['actions'] += 1
        if deal_done:
            self.trick = []
            self.passed_cards = []
            self.log['deals'].append({
                'round_scores': list(map(int, round_scores)),
                'totals': list(map(int, self.menv.match_scores)),
            })
            self.deal_no += 1
        if match_done:
            self.finished = True
            self.log['final'] = list(map(int, self.menv.match_scores))
            self.log['placements'] = list(self.menv.placements())
            self.log['end'] = time.time()
            with open(LOG_PATH, 'a') as f:
                f.write(json.dumps(self.log) + '\n')

    def run_ai_turns(self):
        while (not self.finished
               and self.menv.get_current_player() != self.human_seat):
            self._apply(self.menv.get_current_player(), self._ai_action())

    # -- human-facing state -------------------------------------------------
    def state(self):
        obs = np.asarray(self.menv.env.observe(), dtype=np.float32) \
            if self.menv.get_current_player() == self.human_seat else None
        my_turn = obs is not None and not self.finished
        hand = [int(c) for c in np.flatnonzero(obs[:52] > 0)] if my_turn else []
        legal = self._legal() if my_turn else []
        return {
            'sid': self.sid,
            'finished': self.finished,
            'your_seat': self.human_seat,
            'your_turn': my_turn,
            'passing': bool(self.menv.is_passing()) if my_turn else False,
            'passed_so_far': [card_name(c) for c in self.passed_cards],
            'deal_no': self.deal_no,
            'match_scores': list(map(int, self.menv.match_scores)),
            'hand': [{'card': c, 'name': card_name(c)} for c in sorted(hand)],
            'legal': sorted(legal),
            'trick': [{'seat': s, 'name': card_name(c)} for s, c in self.trick],
            'placements': list(self.menv.placements()) if self.finished else None,
            'target': TARGET,
        }


class PlayBody(BaseModel):
    card: int


@app.post('/api/new')
def new_session():
    s = Session()
    with _sessions_lock:
        _sessions[s.sid] = s
        # Drop oldest sessions past a sane cap
        while len(_sessions) > 500:
            _sessions.pop(next(iter(_sessions)))
    with s.lock:
        s.run_ai_turns()
        return s.state()


def _get(sid):
    with _sessions_lock:
        s = _sessions.get(sid)
    if s is None:
        raise HTTPException(404, 'unknown session')
    return s


@app.get('/api/state/{sid}')
def get_state(sid: str):
    s = _get(sid)
    with s.lock:
        return s.state()


@app.post('/api/play/{sid}')
def play(sid: str, body: PlayBody):
    s = _get(sid)
    with s.lock:
        if s.finished:
            raise HTTPException(409, 'match is over')
        if s.menv.get_current_player() != s.human_seat:
            raise HTTPException(409, 'not your turn')
        if body.card not in s._legal():
            raise HTTPException(400, f'illegal card {body.card}')
        if s.menv.is_passing():
            s.passed_cards.append(body.card)
        s._apply(s.human_seat, body.card)
        s.run_ai_turns()
        return s.state()
