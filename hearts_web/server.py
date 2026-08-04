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
from collections import deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
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


# ---------------------------------------------------------------------------
# Per-IP rate limiting for TUNNELED traffic (requests carrying
# CF-Connecting-IP). Local/LAN requests are exempt - dev flows and phone
# testing never fight the limiter; public traffic always has the header.
# General: 80 requests / 10s / IP (a 4-player NAT household polling at
# 1.5s plus actions is ~30). Creation (session/table/join): 12 / 60s / IP
# - also throttles table-code enumeration.
# ---------------------------------------------------------------------------
_rl_lock = threading.Lock()
_rl_general, _rl_create = {}, {}
RL_GENERAL = (80, 10.0)
RL_CREATE = (12, 60.0)
CREATE_PATHS = ('/api/new', '/api/table/new', '/api/table/join')


def _limited(bucket, ip, limit, window):
    now = time.time()
    with _rl_lock:
        dq = bucket.setdefault(ip, deque())
        while dq and now - dq[0] > window:
            dq.popleft()
        if len(dq) >= limit:
            return True
        dq.append(now)
        if len(bucket) > 10000:      # evict stale IPs, keep memory bounded
            for k in list(bucket):
                d = bucket[k]
                if not d or now - d[-1] > window:
                    bucket.pop(k, None)
    return False


@app.middleware('http')
async def rate_limit(request: Request, call_next):
    ip = request.headers.get('cf-connecting-ip')
    if ip and request.url.path.startswith('/api/'):
        if _limited(_rl_general, ip, *RL_GENERAL):
            return JSONResponse({'detail': 'rate limited'}, status_code=429)
        if request.url.path in CREATE_PATHS and _limited(_rl_create, ip, *RL_CREATE):
            return JSONResponse({'detail': 'rate limited - slow down'},
                                status_code=429)
    return await call_next(request)


def card_name(idx):
    return RANKS[idx % 13] + SUITS[idx // 13]


def ai_action(menv):
    """Argmax raw-net action for the current player of any MatchEnv."""
    obs = torch.from_numpy(menv.observe()).unsqueeze(0)
    mask = torch.zeros((1, 52), dtype=torch.bool)
    for a in menv.get_legal_actions():
        if a != -1:
            mask[0, a] = True
    with _net_lock, torch.no_grad():
        logits, _ = _net(obs, mask)
    return int(torch.argmax(logits, dim=1).item())


def hand_of(menv, seat):
    """Any seat's current hand, server-side ground truth (never shipped to
    anyone but that seat). Rows of observe_opponent_hands are the hands of
    seats (current+k)%4 for k=1..3 (HeartsEnv.ObserveOpponentHandsFor)."""
    p = menv.get_current_player()
    if seat == p:
        obs = np.asarray(menv.env.observe(), dtype=np.float32)
        return sorted(int(c) for c in np.flatnonzero(obs[:52] > 0))
    labels = np.asarray(menv.env.observe_opponent_hands(), dtype=np.float32)
    k = (seat - p) % 4
    return sorted(int(c) for c in np.flatnonzero(labels[(k - 1) * 52:k * 52] > 0))


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
        return ai_action(self.menv)

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


# ---------------------------------------------------------------------------
# TABLE MODE: 1-4 humans + AI fill, join-code lobby, poll-based updates.
# Privacy: each pid is served ONLY its own seat's hand/legality; events carry
# public information only (plays, trick/deal outcomes - never hands/passes).
# ---------------------------------------------------------------------------
CODE_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789'  # no 0/O/1/I/L lookalikes
_tables = {}
_tables_lock = threading.Lock()


class Table:
    def __init__(self, host_pid, host_name):
        self.code = ''.join(secrets.choice(CODE_ALPHABET) for _ in range(4))
        self.state = 'lobby'
        self.lobby = [{'pid': host_pid, 'name': host_name}]  # join order
        self.host_pid = host_pid
        self.seat_of = {}         # pid -> seat (assigned at start)
        self.names = {}           # seat -> display name (humans)
        self.menv = None
        self.seed = None
        self.events = []          # public event log; index == cursor
        self.pending_pass = {}    # seat -> queued cards (applied in env order)
        self.passed_by = {}       # seat -> cards applied this deal
        self.deal_start_hands = {}
        self.received = {}        # seat -> cards received this deal
        self.trick = []
        self.last_trick = None
        self.deal_no = 1
        self.deal_actions = []
        self.n_actions = 0
        self.finished = False
        self.t0 = time.time()
        self.created = time.time()
        self.last_seen = {host_pid: time.time()}   # pid -> last poll (heartbeat)
        self.departed = set()                      # pids that explicitly left
        self.lock = threading.Lock()

    def human_pids(self):
        return ([p['pid'] for p in self.lobby] if self.state == 'lobby'
                else list(self.seat_of))

    # -- lifecycle ----------------------------------------------------------
    def emit(self, type_, **kw):
        self.events.append({'type': type_, **kw})

    def start(self):
        self.state = 'playing'
        self.seed = secrets.randbits(31)
        self.menv = MatchEnv(seed=self.seed)
        seats = list(range(4))
        # Deterministic shuffle from the table seed: fair + reproducible.
        rng = np.random.default_rng(self.seed)
        rng.shuffle(seats)
        for i, p in enumerate(self.lobby):
            self.seat_of[p['pid']] = seats[i]
            self.names[seats[i]] = p['name']
        self._snapshot_deal()
        self.emit('start')
        self.advance()

    def roster(self):
        human = {s: self.names.get(s) for s in self.seat_of.values()}
        return [{'seat': s,
                 'type': 'human' if s in human else 'ai',
                 'name': human.get(s) or f'Player {s + 1} (AI)'}
                for s in range(4)]

    def _snapshot_deal(self):
        self.deal_start_hands = {s: set(hand_of(self.menv, s)) for s in range(4)}
        self.passed_by, self.received, self.pending_pass = {}, {}, {}

    # -- engine pump --------------------------------------------------------
    def humans(self):
        return set(self.seat_of.values())

    def advance(self):
        """Run AI turns and queued human passes until a human must act."""
        while not self.finished:
            s = self.menv.get_current_player()
            if self.menv.is_passing():
                if s in self.humans():
                    q = self.pending_pass.get(s)
                    if not q:
                        break
                    card = q.pop(0)
                else:
                    card = ai_action(self.menv)
                self.passed_by.setdefault(s, []).append(card)
                self._apply(s, card)
            else:
                if s in self.humans():
                    break
                self._apply(s, ai_action(self.menv))

    def _apply(self, seat, action):
        was_passing = self.menv.is_passing()
        self.deal_actions.append((seat, int(action),
                                  int((time.time() - self.t0) * 1000)))
        self.n_actions += 1
        if not was_passing:
            self.trick.append((seat, action))
            self.emit('play', seat=seat, name=card_name(action))
        deal_done, match_done, round_scores = self.menv.step(action)
        if was_passing and not self.menv.is_passing() and not deal_done:
            # Pass exchange resolved: received = hand now minus what was
            # kept (convention-free - no direction mapping needed).
            for s in range(4):
                kept = self.deal_start_hands[s] - set(self.passed_by.get(s, []))
                self.received[s] = sorted(set(hand_of(self.menv, s)) - kept)
            self.emit('passing_done')
        if not was_passing and len(self.trick) == 4:
            winner = None if deal_done else self.menv.get_current_player()
            self.last_trick = {'cards': list(self.trick), 'winner': winner}
            self.emit('trick_end', winner=winner,
                      cards=[{'seat': s, 'name': card_name(c)}
                             for s, c in self.trick])
            self.trick = []
        if deal_done:
            self.trick = []
            srt = sorted(round_scores)
            self.emit('deal_end',
                      round_scores=list(map(int, round_scores)),
                      totals=list(map(int, self.menv.match_scores)),
                      moon_by=(int(np.argmin(round_scores))
                               if srt[0] == 0 and all(v == 26 for v in srt[1:])
                               else None))
            log_line({'v': LOG_V, 'kind': 'deal', 'sid': f'table:{self.code}',
                      'pid': None, 'mode': 'table', 'seed': self.seed,
                      'human_seat': None,
                      'seats': {str(s): ('human' if s in self.humans() else 'ai')
                                for s in range(4)},
                      'model': MODEL_MD5, 'ts': round(time.time(), 3),
                      'deal_no': self.deal_no, 'actions': self.deal_actions,
                      'round_scores': list(map(int, round_scores)),
                      'totals': list(map(int, self.menv.match_scores))})
            self.deal_actions = []
            self.deal_no += 1
            self._snapshot_deal()
        if match_done:
            self.finished = True
            self.emit('match_end',
                      final=list(map(int, self.menv.match_scores)),
                      placements=list(self.menv.placements()))
            log_line({'v': LOG_V, 'kind': 'match', 'sid': f'table:{self.code}',
                      'pid': None, 'mode': 'table', 'seed': self.seed,
                      'human_seat': None,
                      'seats': {str(s): ('human' if s in self.humans() else 'ai')
                                for s in range(4)},
                      'model': MODEL_MD5, 'ts': round(time.time(), 3),
                      'deals': self.deal_no - 1, 'n_actions': self.n_actions,
                      'final': list(map(int, self.menv.match_scores)),
                      'placements': list(self.menv.placements()),
                      'duration_s': round(time.time() - self.t0, 1), 'ua': ''})

    # -- per-seat view ------------------------------------------------------
    def view(self, pid, cursor=0):
        # Every poll is a heartbeat; polling again also revokes a departure
        # (e.g. an accidental leave followed by a rejoin).
        self.last_seen[pid] = time.time()
        self.departed.discard(pid)
        base = {'code': self.code, 'state': self.state, 'target': TARGET}
        if self.state == 'lobby':
            if pid not in (p['pid'] for p in self.lobby):
                raise HTTPException(403, 'not seated at this table')
            return {**base,
                    'players': [p['name'] for p in self.lobby],
                    'you_host': pid == self.host_pid}
        seat = self.seat_of.get(pid)
        if seat is None:
            raise HTTPException(403, 'not seated at this table')
        cur = self.menv.get_current_player() if not self.finished else None
        passing = bool(self.menv.is_passing()) if not self.finished else False
        need_pass = (passing and seat in self.humans()
                     and len(self.passed_by.get(seat, [])) == 0
                     and seat not in self.pending_pass)
        my_turn = (not self.finished and not passing and cur == seat)
        legal = ([a for a in self.menv.get_legal_actions() if a != -1]
                 if my_turn else [])
        waiting = [self.names[s] for s in self.humans()
                   if passing and len(self.passed_by.get(s, [])) == 0
                   and s not in self.pending_pass and s != seat]
        try:
            pass_dir = ['left', 'right', 'across', 'hold'][
                int(self.menv.env.get_pass_direction())]
        except Exception:
            pass_dir = None
        return {**base,
                'you_host': pid == self.host_pid,
                'your_seat': seat, 'roster': self.roster(),
                'your_turn': my_turn, 'passing': need_pass,
                'passed_so_far': [],
                'current_seat': cur, 'waiting_pass': waiting,
                'deal_no': self.deal_no, 'pass_direction': pass_dir,
                'received': [card_name(c) for c in self.received.get(seat, [])],
                'round_scores': list(map(int, self.menv.env.get_round_scores())),
                'match_scores': list(map(int, self.menv.match_scores)),
                'hand': [{'card': c, 'name': card_name(c)}
                         for c in hand_of(self.menv, seat)],
                'legal': sorted(legal),
                'trick': [{'seat': s, 'name': card_name(c)}
                          for s, c in self.trick],
                'last_trick': (None if self.last_trick is None else {
                    'cards': [{'seat': s, 'name': card_name(c)}
                              for s, c in self.last_trick['cards']],
                    'winner': self.last_trick['winner']}),
                'finished': self.finished,
                'placements': (list(self.menv.placements())
                               if self.finished else None),
                'events': self.events[cursor:], 'cursor': len(self.events)}


def _get_table(code):
    with _tables_lock:
        t = _tables.get(code.upper())
    if t is None:
        raise HTTPException(404, 'unknown table code')
    return t


class PlayBody(BaseModel):
    card: int


class NewBody(BaseModel):
    pid: str | None = None


@app.get('/')
def index(request: Request):
    """Serve the app; dev controls (reset button, ?player= identity
    override) are injected ONLY for direct localhost requests - anything
    arriving through the tunnel (CF-Connecting-IP present) or the LAN
    gets DEV_CONTROLS = false."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'static', 'index.html')
    with open(path, encoding='utf-8') as f:
        html = f.read()
    local = (request.headers.get('cf-connecting-ip') is None
             and request.client is not None
             and request.client.host in ('127.0.0.1', '::1'))
    if not local:
        html = html.replace('const DEV_CONTROLS = true;',
                            'const DEV_CONTROLS = false;', 1)
    return Response(content=html, media_type='text/html')


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


class TableNewBody(BaseModel):
    pid: str
    name: str | None = None


class TableJoinBody(BaseModel):
    code: str
    pid: str
    name: str | None = None


class TableActBody(BaseModel):
    pid: str
    card: int | None = None
    cards: list[int] | None = None
    cursor: int = 0


def _clean_name(name, fallback):
    # Strip HTML-significant chars server-side (client escapes too -
    # belt and braces; names are the only user strings ever displayed).
    n = ''.join(c for c in (name or '') if c not in '<>&"\'').strip()[:20]
    return n if n else fallback


@app.post('/api/table/new')
def table_new(body: TableNewBody):
    t = Table(body.pid[:64], _clean_name(body.name, 'Host'))
    with _tables_lock:
        while t.code in _tables:
            t.code = ''.join(secrets.choice(CODE_ALPHABET) for _ in range(4))
        # Drop stale tables past a sane cap (oldest first).
        while len(_tables) > 200:
            _tables.pop(next(iter(_tables)))
        _tables[t.code] = t
    return t.view(body.pid)


@app.post('/api/table/join')
def table_join(body: TableJoinBody):
    t = _get_table(body.code)
    with t.lock:
        pid = body.pid[:64]
        if any(p['pid'] == pid for p in t.lobby) or pid in t.seat_of:
            return t.view(pid)          # idempotent rejoin
        if t.state != 'lobby':
            raise HTTPException(409, 'match already in progress - ask the '
                                     'host to close and start a new table')
        if len(t.lobby) >= 4:
            raise HTTPException(409, 'table is full (4 players max)')
        n = _clean_name(body.name, f'Guest {len(t.lobby) + 1}')
        t.lobby.append({'pid': pid, 'name': n})
        return t.view(pid)


@app.post('/api/table/start')
def table_start(body: TableJoinBody):
    t = _get_table(body.code)
    with t.lock:
        if body.pid != t.host_pid:
            raise HTTPException(403, 'only the host can start')
        if t.state == 'lobby':
            t.start()
        return t.view(body.pid)


@app.get('/api/table/state/{code}')
def table_state(code: str, pid: str, cursor: int = 0):
    t = _get_table(code)
    with t.lock:
        return t.view(pid, cursor)


@app.post('/api/table/close')
def table_close(body: TableJoinBody):
    """Host-only: delete the table for everyone (end-game screen button).
    Other players' next poll gets 404 and returns to the menu cleanly."""
    t = _get_table(body.code)
    with t.lock:
        if body.pid != t.host_pid:
            raise HTTPException(403, 'only the host can close the table')
    with _tables_lock:
        _tables.pop(t.code, None)
    return {'ok': True}


@app.post('/api/table/leave')
def table_leave(body: TableJoinBody):
    """Explicit leave (menu/reset button). When the LAST human leaves this
    way the table is deleted immediately; silent departures (closed tabs,
    dead links) are reaped by staleness instead."""
    t = _get_table(body.code)
    with t.lock:
        pid = body.pid[:64]
        t.departed.add(pid)
        if t.state == 'lobby':
            if pid == t.host_pid:
                all_gone = True          # host abandoned the lobby: close it
            else:
                t.lobby = [p for p in t.lobby if p['pid'] != pid]
                all_gone = not t.lobby
        else:
            all_gone = all(p in t.departed for p in t.human_pids())
    if all_gone:
        with _tables_lock:
            _tables.pop(t.code, None)
    return {'ok': True, 'closed': all_gone}


def _reaper():
    """Delete tables whose humans are all gone: explicitly departed, or
    silent for over 2 minutes (the 1.5s poll is the heartbeat - a refresh
    takes seconds and can never trip this)."""
    while True:
        time.sleep(60)
        now = time.time()
        with _tables_lock:
            items = list(_tables.items())
        for code, t in items:
            with t.lock:
                pids = t.human_pids()
                drop = (not pids or
                        all(p in t.departed
                            or now - t.last_seen.get(p, t.created) > 120
                            for p in pids))
            if drop:
                with _tables_lock:
                    _tables.pop(code, None)


threading.Thread(target=_reaper, daemon=True).start()


@app.post('/api/table/pass/{code}')
def table_pass(code: str, body: TableActBody):
    t = _get_table(code)
    with t.lock:
        seat = t.seat_of.get(body.pid)
        if seat is None:
            raise HTTPException(403, 'not seated at this table')
        if t.state != 'playing' or t.finished or not t.menv.is_passing():
            raise HTTPException(409, 'not in a passing phase')
        if t.passed_by.get(seat) or seat in t.pending_pass:
            raise HTTPException(409, 'already passed')
        cards = body.cards or []
        hand = set(hand_of(t.menv, seat))
        if len(cards) != 3 or len(set(cards)) != 3 or not set(cards) <= hand:
            raise HTTPException(400, 'pass must be 3 distinct cards from your hand')
        t.pending_pass[seat] = list(cards)
        t.advance()
        return t.view(body.pid, body.cursor)


@app.post('/api/table/play/{code}')
def table_play(code: str, body: TableActBody):
    t = _get_table(code)
    with t.lock:
        seat = t.seat_of.get(body.pid)
        if seat is None:
            raise HTTPException(403, 'not seated at this table')
        if t.state != 'playing' or t.finished:
            raise HTTPException(409, 'table not in play')
        if t.menv.is_passing() or t.menv.get_current_player() != seat:
            raise HTTPException(409, 'not your turn')
        if body.card not in [a for a in t.menv.get_legal_actions() if a != -1]:
            raise HTTPException(400, f'illegal card {body.card}')
        t._apply(seat, body.card)
        t.advance()
        return t.view(body.pid, body.cursor)


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
