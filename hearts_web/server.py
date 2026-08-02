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
import hashlib
import json
import os
import secrets
import sys
import threading
import time

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hearts_match_env import MatchEnv, TARGET  # noqa: E402
from hearts_net import net_from_checkpoint  # noqa: E402

SUITS = 'CDSH'
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'match_logs.jsonl')

app = FastAPI(title="Hearts vs AI")

MODEL_PATH = 'hearts_model_final.pth'
_net = net_from_checkpoint(MODEL_PATH)
_net.eval()
_net_lock = threading.Lock()
_sessions = {}
_sessions_lock = threading.Lock()

# Identity stamp for every log line: which weights produced the AI moves.
with open(MODEL_PATH, 'rb') as _f:
    MODEL_MD5 = hashlib.md5(_f.read()).hexdigest()[:12]
LOG_V = 2
_log_lock = threading.Lock()


def log_line(obj):
    with _log_lock, open(LOG_PATH, 'a') as f:
        f.write(json.dumps(obj) + '\n')


def card_name(idx):
    return RANKS[idx % 13] + SUITS[idx // 13]


class Session:
    def __init__(self, pid=None, ua=''):
        self.sid = secrets.token_urlsafe(12)
        self.seed = secrets.randbits(31)
        self.menv = MatchEnv(seed=self.seed)
        self.human_seat = secrets.randbelow(4)
        self.pid = pid            # anonymous persistent player id (client)
        self.ua = (ua or '')[:120]
        self.trick = []           # [(seat, card)] of the current trick
        self.last_trick = None    # {'cards': [(seat, card)], 'winner': seat}
        self.last_deal = None     # round scores of the most recent deal
        self.deal_no = 1
        self.passed_cards = []    # human's picks this pass phase
        self.events = []          # ordered happenings since the last human action
        self.t0 = time.time()
        self.deal_actions = []    # [(seat, card, ms_since_match_start)]
        self.n_actions = 0
        self.finished = False
        self.lock = threading.Lock()

    def _stamp(self, kind):
        return {'v': LOG_V, 'kind': kind, 'sid': self.sid, 'pid': self.pid,
                'seed': self.seed, 'human_seat': self.human_seat,
                'model': MODEL_MD5, 'ts': round(time.time(), 3)}

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
        in_play = not self.menv.is_passing()
        # Full per-card record (replay contract: MatchEnv(seed) + this
        # action sequence reproduces the match bit-exactly). ms is server
        # arrival time - for human actions an upper bound on think time
        # (client animations inflate it).
        self.deal_actions.append((seat, int(action),
                                  int((time.time() - self.t0) * 1000)))
        self.n_actions += 1
        if in_play:
            self.trick.append((seat, action))
            self.events.append({'type': 'play', 'seat': seat,
                                'name': card_name(action)})
        deal_done, match_done, round_scores = self.menv.step(action)
        if in_play and len(self.trick) == 4:
            # The next current player is the trick winner (they lead next),
            # unless the deal just ended (then round_scores tell the story)
            winner = None if deal_done else self.menv.get_current_player()
            self.last_trick = {'cards': list(self.trick), 'winner': winner}
            self.events.append({'type': 'trick_end', 'winner': winner,
                                'cards': [{'seat': s, 'name': card_name(c)}
                                          for s, c in self.last_trick['cards']]})
            self.trick = []
        if deal_done:
            self.trick = []
            self.last_deal = list(map(int, round_scores))
            srt = sorted(round_scores)
            self.events.append({
                'type': 'deal_end',
                'round_scores': list(map(int, round_scores)),
                'totals': list(map(int, self.menv.match_scores)),
                'moon_by': (int(np.argmin(round_scores))
                            if srt[0] == 0 and all(v == 26 for v in srt[1:])
                            else None)})
            self.passed_cards = []
            # Flush one line per completed deal: abandoned matches keep
            # every finished deal (only the in-progress one is lost).
            log_line({**self._stamp('deal'), 'deal_no': self.deal_no,
                      'actions': self.deal_actions,
                      'round_scores': list(map(int, round_scores)),
                      'totals': list(map(int, self.menv.match_scores))})
            self.deal_actions = []
            self.deal_no += 1
        if match_done:
            self.finished = True
            log_line({**self._stamp('match'), 'deals': self.deal_no - 1,
                      'n_actions': self.n_actions,
                      'final': list(map(int, self.menv.match_scores)),
                      'placements': list(self.menv.placements()),
                      'duration_s': round(time.time() - self.t0, 1),
                      'ua': self.ua})

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
        # Obs block 9 (238-289): cards received in this deal's pass
        received = ([card_name(int(c)) for c in np.flatnonzero(obs[238:290] > 0)]
                    if my_turn else [])
        try:
            pass_dir = ['left', 'right', 'across', 'hold'][
                int(self.menv.env.get_pass_direction())]
        except Exception:
            pass_dir = None
        return {
            'sid': self.sid,
            'finished': self.finished,
            'your_seat': self.human_seat,
            'your_turn': my_turn,
            'passing': bool(self.menv.is_passing()) if my_turn else False,
            'passed_so_far': [card_name(c) for c in self.passed_cards],
            'deal_no': self.deal_no,
            'pass_direction': pass_dir,
            'received': received,
            'round_scores': list(map(int, self.menv.env.get_round_scores())),
            'match_scores': list(map(int, self.menv.match_scores)),
            'hand': [{'card': c, 'name': card_name(c)} for c in sorted(hand)],
            'legal': sorted(legal),
            'trick': [{'seat': s, 'name': card_name(c)} for s, c in self.trick],
            'last_trick': (None if self.last_trick is None else {
                'cards': [{'seat': s, 'name': card_name(c)}
                          for s, c in self.last_trick['cards']],
                'winner': self.last_trick['winner']}),
            'last_deal': self.last_deal,
            'placements': list(self.menv.placements()) if self.finished else None,
            'target': TARGET,
        }


class PlayBody(BaseModel):
    card: int


class NewBody(BaseModel):
    pid: str | None = None


@app.get('/')
def index():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'static', 'index.html'))


@app.post('/api/new')
def new_session(body: NewBody | None = None, request: Request = None):
    pid = (body.pid[:64] if body and body.pid else None)
    ua = request.headers.get('user-agent', '') if request else ''
    s = Session(pid=pid, ua=ua)
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
        s.events = []
        if s.menv.is_passing():
            s.passed_cards.append(body.card)
        s._apply(s.human_seat, body.card)
        s.run_ai_turns()
        out = s.state()
        out['events'] = s.events
        return out
