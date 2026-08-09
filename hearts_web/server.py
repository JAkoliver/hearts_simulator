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
import base64
import hashlib
import hmac
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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hearts_match_env import MatchEnv, TARGET  # noqa: E402
from hearts_net import net_from_checkpoint  # noqa: E402

SUITS = 'CDSH'
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'match_logs.jsonl')

app = FastAPI(title="Perilune - Hearts vs AI")
app.mount('/static', StaticFiles(directory=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'static')), name='static')

# Site-ops config: local site_config.py (gitignored, production values)
# with the committed example as fallback - publish the instrument, keep
# the operations (release-boundary decision 2026-08-04).
import importlib.util as _ilu
_cfg_dir = os.path.dirname(os.path.abspath(__file__))
_cfg_path = os.path.join(_cfg_dir, 'site_config.py')
if not os.path.exists(_cfg_path):
    _cfg_path = os.path.join(_cfg_dir, 'site_config_example.py')
_spec = _ilu.spec_from_file_location('site_config', _cfg_path)
cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

MODEL_PATH = cfg.MODEL_PATH
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


# ---------------------------------------------------------------------------
# Difficulty tiers: which net plays the AI seats. Easier tiers are the
# frozen earlier-generation anchors the research gates measure against
# (v3-m7, v4-m10 traces); 'full' is the current baseline. Labels carry the
# model generation so the opponent is never mystery-branded. Review and
# insight evals ALWAYS use the full-strength net regardless of play tier.
# ---------------------------------------------------------------------------
class _TierNet:
    def __init__(self, module, in_dim):
        self.module, self.in_dim = module, in_dim

    def act(self, obs, mask):
        with _net_lock, torch.no_grad():
            out = self.module(obs[:, :self.in_dim], mask)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        return int(torch.argmax(logits, dim=1).item())


def _load_tiers():
    tiers = {'full': {'label': 'Perilune · v5 (full strength)',
                      'net': _TierNet(_net, 556), 'md5': MODEL_MD5}}
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    for key, label, rel, dim in getattr(cfg, 'TIERS', [
            ('casual', 'Casual · v3',
             os.path.join('legacy_v3_pass238',
                          'hearts_ai_grandmaster_v3_milestone7.pt'), 238),
            ('standard', 'Standard · v4',
             'hearts_ai_grandmaster_v4m10.pt', 550)]):
        path = rel if os.path.isabs(rel) else os.path.join(root, rel)
        try:
            m = torch.jit.load(path)
            m.eval()
            with open(path, 'rb') as f:
                md5 = hashlib.md5(f.read()).hexdigest()[:12]
            tiers[key] = {'label': label, 'net': _TierNet(m, dim), 'md5': md5}
        except Exception as e:   # tier nets are optional; 'full' always works
            print(f'[tiers] {key} unavailable: {e}')
    return tiers


TIERS = _load_tiers()
TIER_ORDER = [k for k in ('casual', 'standard', 'full') if k in TIERS]
TURN_TIMER_S = getattr(cfg, 'TURN_TIMER_S', 60)   # table AFK timer; 0 = off


def _norm_tier(t):
    return t if t in TIERS else 'full'


# ---------------------------------------------------------------------------
# Log index (in-memory, exact): this process is the log's only writer, so a
# one-pass build at startup plus an update on every append keeps review /
# insight / history from rescanning the whole JSONL per request.
#   _idx_deals    sid -> [byte offset of each deal line]  (seek-and-read)
#   _idx_seatpids (sid, match_no) -> first seat_pids seen (old table matches
#                 logged summaries before seat_pids existed on them)
#   _idx_history  pid -> [ready history entries, append order]
# Offsets are BINARY-file offsets; the log is appended in binary from here on
# (legacy text-mode lines ended \r\n - readers tolerate both).
# ---------------------------------------------------------------------------
_idx_deals = {}
_idx_seatpids = {}
_idx_history = {}
# Leaderboard (per MODEL ERA, option-b design): era = the model md5 the
# match was played against; each era keeps every player's single best
# SOLE solo win vs the full tier. Archived eras stay browsable forever.
_lb = {}          # era(md5[:12]) -> {canonical_pid: entry}
# Matches with a logged summary line - i.e. FINISHED. Reviews/insights/
# share-minting gate on this: the review payload carries the match SEED,
# and one seed drives every future deal of the match, so a mid-match
# review would let a player simulate all remaining hands.
_finished_matches = set()   # (sid, match_no)


def _index_add(d, off):
    if d.get('v') != LOG_V:
        return
    key = (d.get('sid'), d.get('match_no', 1))
    kind = d.get('kind')
    if kind == 'deal':
        _idx_deals.setdefault(d['sid'], []).append(off)
        if d.get('seat_pids') and key not in _idx_seatpids:
            _idx_seatpids[key] = d['seat_pids']
    elif kind == 'match':
        _finished_matches.add((d['sid'], d.get('match_no', 1)))
        if d.get('mode') == 'table':
            sp = d.get('seat_pids') or _idx_seatpids.get(key, {})
            for s, p in sp.items():
                _idx_history.setdefault(p, []).append(
                    {'mode': 'table', 'code': d['sid'].split(':', 1)[1],
                     'match_no': d.get('match_no', 1), 'ts': d['ts'],
                     'deals': d['deals'], 'seat': int(s),
                     'place': d['placements'][int(s)], 'final': d['final']})
        elif d.get('pid'):
            seat = d['human_seat']
            _idx_history.setdefault(d['pid'], []).append(
                {'mode': 'solo', 'sid': d['sid'], 'ts': d['ts'],
                 'deals': d['deals'], 'seat': seat,
                 'place': d['placements'][seat], 'final': d['final']})
            # Leaderboard: SOLE first place (tied placements are floats,
            # so == 1 excludes them), solo, full tier only (tier absent =
            # the pre-tier era, which was all full-strength).
            if (d.get('tier') in (None, 'full')
                    and d['placements'][seat] == 1):
                era = (d.get('model') or 'unknown')[:12]
                score = int(d['final'][seat])
                e = _lb.setdefault(era, {})
                cur = e.get(d['pid'])
                # strictly-better replaces; equal keeps the EARLIER win
                if cur is None or score < cur['score']:
                    e[d['pid']] = {'canon': d['pid'], 'score': score,
                                   'deals': d['deals'], 'ts': d['ts'],
                                   'sid': d['sid'], 'seat': seat}


def _build_log_index():
    try:
        with open(LOG_PATH, 'rb') as f:
            off = 0
            for raw in f:
                if raw.strip():
                    try:
                        _index_add(json.loads(raw.decode()), off)
                    except (ValueError, KeyError):
                        pass
                off += len(raw)
    except FileNotFoundError:
        pass


_build_log_index()


def log_line(obj):
    with _log_lock:
        with open(LOG_PATH, 'ab') as f:
            off = f.seek(0, 2)
            f.write((json.dumps(obj) + '\n').encode())
        try:
            _index_add(obj, off)
        except KeyError:
            pass   # a malformed entry must never block the log append


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
RL_GENERAL = cfg.RL_GENERAL
RL_CREATE = cfg.RL_CREATE
CREATE_PATHS = ('/api/new', '/api/table/new', '/api/table/join',
                '/api/identity/new', '/api/identity/rotate')


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


def ai_action(menv, tier='full'):
    """Argmax action for the current player, from the tier's net."""
    obs = torch.from_numpy(menv.observe()).unsqueeze(0)
    mask = torch.zeros((1, 52), dtype=torch.bool)
    for a in menv.get_legal_actions():
        if a != -1:
            mask[0, a] = True
    return TIERS[_norm_tier(tier)]['net'].act(obs, mask)


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
    def __init__(self, pid=None, ua='', tier='full'):
        self.sid = secrets.token_urlsafe(12)
        self.seed = secrets.randbits(31)
        self.menv = MatchEnv(seed=self.seed)
        self.human_seat = secrets.randbelow(4)
        self.pid = pid            # anonymous persistent player id (client)
        self.tier = _norm_tier(tier)
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
                'model': TIERS[self.tier]['md5'], 'tier': self.tier,
                'ts': round(time.time(), 3)}

    # -- engine helpers -----------------------------------------------------
    def _legal(self):
        return [a for a in self.menv.get_legal_actions() if a != -1]

    def _ai_action(self):
        return ai_action(self.menv, self.tier)

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
            'tier': self.tier,
            'tier_label': TIERS[self.tier]['label'],
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
    def __init__(self, host_pid, host_name, tier='full'):
        self.code = ''.join(secrets.choice(CODE_ALPHABET)
                            for _ in range(cfg.CODE_LEN))
        self.tier = _norm_tier(tier)
        self.timer_s = TURN_TIMER_S   # host-set at start (0 = no timer)
        self.turn_deadline = None   # AFK timer for the blocking human seat
        self.timeouts = []          # indices into deal_actions auto-played
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
        self.match_no = 1          # increments on rematch (client epoch)
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

    def rematch(self):
        """Fresh match at the same table: same code, same seats (minus
        departed players, whose seats revert to AI), new seed/deal."""
        for pid in list(self.seat_of):
            if pid in self.departed:
                self.names.pop(self.seat_of[pid], None)
                del self.seat_of[pid]
        self.match_no += 1
        self.seed = secrets.randbits(31)
        self.menv = MatchEnv(seed=self.seed)
        self.events = []
        self.trick, self.last_trick = [], None
        self.deal_no, self.deal_actions, self.n_actions = 1, [], 0
        self.finished = False
        self.t0 = time.time()
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
        self.timeouts = []

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
                    card = ai_action(self.menv, self.tier)
                self.passed_by.setdefault(s, []).append(card)
                self._apply(s, card)
            else:
                if s in self.humans():
                    break
                self._apply(s, ai_action(self.menv, self.tier))
        # Arm the AFK timer for whichever human we stopped on.
        if (self.timer_s and self.state == 'playing' and not self.finished
                and self.menv.get_current_player() in self.humans()):
            self.turn_deadline = time.time() + self.timer_s
        else:
            self.turn_deadline = None

    def check_timeout(self):
        """AFK enforcement, called from every state poll. The auto-play is a
        deliberately DUMB heuristic (lowest card of the current suit /
        lowest 3 for a pass) so waiting the timer out is never a way to
        make the strong AI play for you."""
        if (not self.timer_s or self.state != 'playing' or self.finished
                or self.turn_deadline is None
                or time.time() < self.turn_deadline):
            return
        s = self.menv.get_current_player()
        if s not in self.humans():
            self.turn_deadline = None
            return
        legal = [a for a in self.menv.get_legal_actions() if a != -1]
        low = sorted(legal, key=lambda a: (a % 13, a // 13))
        if self.menv.is_passing():
            self.pending_pass[s] = low[:3]
            self.emit('timeout', seat=s, what='pass')
        else:
            self.timeouts.append(len(self.deal_actions))
            self.emit('timeout', seat=s, what='play',
                      name=card_name(low[0]))
            self._apply(s, low[0])
        self.turn_deadline = None
        self.advance()

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
                      'human_seat': None, 'match_no': self.match_no,
                      'seats': {str(s): ('human' if s in self.humans() else 'ai')
                                for s in range(4)},
                      'seat_pids': {str(s): p for p, s in self.seat_of.items()},
                      'model': TIERS[self.tier]['md5'], 'tier': self.tier,
                      'timeouts': list(self.timeouts),
                      'ts': round(time.time(), 3),
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
                      'human_seat': None, 'match_no': self.match_no,
                      'seat_pids': {str(s): p for p, s in self.seat_of.items()},
                      'seats': {str(s): ('human' if s in self.humans() else 'ai')
                                for s in range(4)},
                      'model': TIERS[self.tier]['md5'], 'tier': self.tier,
                      'ts': round(time.time(), 3),
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
        base = {'code': self.code, 'state': self.state, 'target': TARGET,
                'tier': self.tier, 'tier_label': TIERS[self.tier]['label']}
        if self.state == 'playing' and self.turn_deadline is not None:
            base['turn_seconds_left'] = max(
                0, int(self.turn_deadline - time.time()))
            base['turn_timer_s'] = self.timer_s
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
        now = time.time()
        away = [s for p, s in self.seat_of.items()
                if p != pid and (p in self.departed
                                 or now - self.last_seen.get(p, 0) > cfg.AWAY_S)]
        return {**base,
                'you_host': pid == self.host_pid,
                'match_no': self.match_no, 'away_seats': sorted(away),
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
    pid: str | None = None


class NewBody(BaseModel):
    pid: str | None = None
    tier: str | None = None


# ---------------------------------------------------------------------------
# Match review (/review): full-match replay payload - every state, every
# seat's hands (x-ray; post-game public), the net's top-3 at every play
# (info-honest: computed from what THAT seat could see), moon-threat
# markers, and a per-deal win-probability strip from the equity net.
# Reads the telemetry log, so it is independent of live tables (rematch/
# close cannot disturb an open review tab).
# ---------------------------------------------------------------------------
_equity = None


def _equity_net():
    global _equity
    if _equity is None:
        try:
            _equity = torch.jit.load('hearts_equity.pt')
            _equity.eval()
        except Exception:
            _equity = False
    return _equity or None


def _win_probs(totals, deals_played):
    """Per-seat P(place 1). Exact tie-split when the match is over, else
    the equity net with the DEPLOYED input layout (SearchPlayer.hpp
    ScoreEquity: rotation, deals/20, leader distance, onehot[deals%4])."""
    totals = [float(t) for t in totals]
    if max(totals) >= 100:
        best = min(totals)
        winners = [i for i, t in enumerate(totals) if t == best]
        return [round(1.0 / len(winners), 4) if i in winners else 0.0
                for i in range(4)]
    net = _equity_net()
    if net is None:
        return None
    x = torch.zeros((4, 10))
    for s in range(4):
        for k in range(4):
            x[s, k] = totals[(s + k) % 4] / 100.0
        x[s, 4] = deals_played / 20.0
        x[s, 5] = (100.0 - max(totals)) / 100.0
        x[s, 6 + (deals_played % 4)] = 1.0
    with torch.no_grad():
        probs = torch.softmax(net(x), 1)
    return [round(float(probs[s, 0]), 4) for s in range(4)]


_QS = SUITS.index('S') * 13 + RANKS.index('Q')


def _equiv_groups(hand_set, played_set, legal):
    """Groups of strictly-equivalent legal cards: same suit, every rank
    between them in the acting seat's own hand or already played
    (visible info only), and equal penalty value. Such cards win/lose
    identical tricks AND score identically in every continuation -
    preferences inside a group are meaningless.

    The QS never joins a group (13 points vs 0 for its neighbors),
    though held it still bridges J-K connectivity. Hearts all carry the
    same 1 point, so heart groups are fine."""
    by_suit = {}
    for c in legal:
        if c == _QS:      # 13 points: never equivalent to 0-point neighbors
            continue
        by_suit.setdefault(c // 13, []).append(c)
    groups = []
    for suit, cards in by_suit.items():
        cards.sort()
        cur = [cards[0]]
        for prev, nxt in zip(cards, cards[1:]):
            ok = all(
                (suit * 13 + r) in hand_set or (suit * 13 + r) in played_set
                for r in range(prev % 13 + 1, nxt % 13))
            if ok:
                cur.append(nxt)
            else:
                groups.append(cur)
                cur = [nxt]
        groups.append(cur)
    return [g for g in groups if len(g) > 1]


_review_cache = {}   # (sid_key, match_no) -> seat-independent payload


def compute_review(deal_lines, viewer_seat):
    # PASS 1: replay once, collecting every play-state observation; then
    # ONE batched net forward (sequential per-play forwards took minutes
    # under load - measured 2026-08-04).
    deal_lines = sorted(deal_lines, key=lambda d: d['deal_no'])
    menv = MatchEnv(seed=deal_lines[0]['seed'])
    out_deals = []
    win0 = _win_probs([0, 0, 0, 0], 0)
    all_obs, all_mask, all_ref = [], [], []   # ref -> (deal_idx, play_idx)
    for di, d in enumerate(deal_lines):
        start_hands = [sorted(hand_of(menv, s)) for s in range(4)]
        try:
            pdir = int(menv.env.get_pass_direction())
        except Exception:
            pdir = 3
        passed = [[] for _ in range(4)]
        plays, pass_evals, threats = [], [], []
        taken = [0, 0, 0, 0]
        trick_cards = []
        played_deal = set()   # cards visibly played this deal (equivalence)
        for s, card, ms in d['actions']:
            if menv.get_current_player() != s:
                raise HTTPException(500, 'replay desync in review')
            if menv.is_passing():
                # Pass picks are decision states too: same batched eval.
                mask = np.zeros(52, dtype=bool)
                legal = []
                for a in menv.get_legal_actions():
                    if a != -1:
                        mask[a] = True
                        legal.append(a)
                obs = np.array(menv.observe(), dtype=np.float32)
                all_obs.append(obs)
                all_mask.append(mask)
                all_ref.append(('pass', di, len(pass_evals)))
                hand = set(np.flatnonzero(obs[:52] > 0).tolist())
                eq = [[card_name(c) for c in g]
                      for g in _equiv_groups(hand, set(), legal)]
                pass_evals.append([s, card_name(card), 0.0, [], eq])
                passed[s].append(card)
                menv.step(card)
                continue
            if not trick_cards:
                # Entering a new trick: moon-threat check on points so far.
                holders = [i for i in range(4) if taken[i] > 0]
                if len(holders) == 1 and taken[holders[0]] >= 4:
                    threats.append({'trick': len(plays) // 4 + 1,
                                    'seat': holders[0]})
            mask = np.zeros(52, dtype=bool)
            legal = []
            for a in menv.get_legal_actions():
                if a != -1:
                    mask[a] = True
                    legal.append(a)
            obs = np.array(menv.observe(), dtype=np.float32)
            all_obs.append(obs)
            all_mask.append(mask)
            all_ref.append(('play', di, len(plays)))
            # Seat-locked belief lens: the other seats' info-honest views
            # of this same position (belief rows only - the mask is the
            # actor's and the policy outputs of these rows are unused).
            for os_ in range(4):
                if os_ != s:
                    all_obs.append(np.array(menv.observe_for(os_),
                                            dtype=np.float32))
                    all_mask.append(mask)
                    all_ref.append(('bel', di, len(plays), os_))
            hand = set(np.flatnonzero(obs[:52] > 0).tolist())
            eq = [[card_name(c) for c in g]
                  for g in _equiv_groups(hand, played_deal, legal)]
            plays.append([s, card_name(card), 0.0, [], eq])  # evals fill below
            played_deal.add(card)
            trick_cards.append((s, card))
            if len(trick_cards) == 4:
                lead = trick_cards[0][1] // 13
                ws, wc = trick_cards[0]
                for ts, tc in trick_cards[1:]:
                    if tc // 13 == lead and tc % 13 > wc % 13:
                        ws, wc = ts, tc
                taken[ws] += sum(1 if c // 13 == 3 else (13 if c == 36 else 0)
                                 for _, c in trick_cards)
                trick_cards = []
            menv.step(card)
        # Received = post-pass hand minus what was kept; the play-phase
        # cards per seat ARE the post-pass hand (all 13 get played).
        received = [[] for _ in range(4)]
        if pdir != 3:
            passing_n = sum(len(p) for p in passed)
            played_sets = {s: set() for s in range(4)}
            for j, (s, card, ms) in enumerate(d['actions']):
                if j >= passing_n:
                    played_sets[s].add(card)
            for s in range(4):
                kept = set(start_hands[s]) - set(passed[s])
                received[s] = sorted(played_sets[s] - kept)
        out_deals.append({
            'deal_no': d['deal_no'],
            'start_hands': [[card_name(c) for c in h] for h in start_hands],
            'pass_direction': ['left', 'right', 'across', 'hold'][pdir],
            'passed': [[card_name(c) for c in p] for p in passed],
            'received': [[card_name(c) for c in r] for r in received],
            'plays': plays, 'pass_evals': pass_evals, 'threats': threats,
            'round_scores': d['round_scores'], 'totals': d['totals'],
            'win_prob_after': _win_probs(d['totals'], d['deal_no'])})
    # PASS 2: single batched forward for every play state.
    if all_obs:
        obs_t = torch.from_numpy(np.stack(all_obs))
        mask_t = torch.from_numpy(np.stack(all_mask))
        chunks = []
        bchunks = []
        with _net_lock, torch.no_grad():
            for i in range(0, len(all_obs), 512):
                logits, _, bel = _net.forward_all(obs_t[i:i + 512],
                                                  mask_t[i:i + 512])
                chunks.append(torch.softmax(logits, dim=1))
                bchunks.append(torch.sigmoid(bel))
        probs = torch.cat(chunks)
        beliefs = torch.cat(bchunks)

        def enc_belief(row):
            return base64.b64encode(
                (beliefs[row] * 255).round().clamp(0, 255)
                .to(torch.uint8).numpy().tobytes()).decode()

        bel_rows = {}   # (di, pi) -> {seat: forward row}
        for row, ref in enumerate(all_ref):
            kind, di, pi = ref[0], ref[1], ref[2]
            if kind == 'bel':
                bel_rows.setdefault((di, pi), {})[ref[3]] = row
                continue
            play = out_deals[di]['plays' if kind == 'play' else 'pass_evals'][pi]
            card = play[1]
            pr = probs[row]
            # Full ranked legal list (client shows top-3 with an expander).
            k = int(mask_t[row].sum())
            topv, topi = torch.topk(pr, k)
            cid = RANKS.index(card[:-1]) + SUITS.index(card[-1]) * 13
            play[2] = round(float(pr[cid]), 3)
            play[3] = [[card_name(int(topi[j])), round(float(topv[j]), 3)]
                       for j in range(k)]
            if kind == 'play':
                bel_rows.setdefault((di, pi), {})[play[0]] = row
            elif kind == 'pass':
                # Pass-phase belief: the passer's own pre-pick view,
                # single b64 at row[5] (plays get the 4-seat list form)
                play.append(enc_belief(row))
        # Belief heatmap (client board layer, seat-lockable): play[5] =
        # per-SEAT belief heads (3 relative opponents x 52 cards each,
        # sigmoid, uint8, base64), absolute seat order 0..3.
        for (di, pi), rows in bel_rows.items():
            out_deals[di]['plays'][pi].append(
                [enc_belief(rows[s_]) for s_ in range(4)])
    return {'viewer_seat': viewer_seat,
            'seat_types': deal_lines[0].get('seats'),
            'win_prob_start': win0, 'deals': out_deals,
            # Client-side deep analysis (WASM engine): the replay contract.
            # Raw action ids per deal INCLUDING pass picks - the engine
            # rebuilds any decision state from (seed, prefix).
            'replay': {'seed': deal_lines[0]['seed'],
                       'deal_actions': [[a for _, a, _ in d['actions']]
                                        for d in deal_lines]},
            'note': ('Evals are information-honest: each seat is judged on '
                     'what it could see, even though the review shows all '
                     'hands.')}


# ---------------------------------------------------------------------------
# Shareable review links: stateless HMAC tokens. A player who was seated in
# a match can mint a read-only link; the token carries the match identity
# and the sharer's seat, so the shared view opens from their perspective
# without any pid. The secret persists across restarts so links stay valid.
# ---------------------------------------------------------------------------
_SHARE_SECRET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '.share_secret')
try:
    with open(_SHARE_SECRET_PATH, 'rb') as _f:
        _SHARE_KEY = _f.read()
    if len(_SHARE_KEY) < 32:
        raise IOError('short secret')
except IOError:
    _SHARE_KEY = secrets.token_bytes(32)
    with open(_SHARE_SECRET_PATH, 'wb') as _f:
        _f.write(_SHARE_KEY)


def _share_sign(payload: str) -> str:
    return hmac.new(_SHARE_KEY, payload.encode(),
                    hashlib.sha256).hexdigest()[:20]


def _share_make(kind, ident, match_no, seat):
    payload = f'{kind}|{ident}|{match_no}|{seat}'
    tok = base64.urlsafe_b64encode(payload.encode()).decode().rstrip('=')
    return f'{tok}.{_share_sign(payload)}'


def _share_parse(token):
    try:
        tok, sig = token.rsplit('.', 1)
        payload = base64.urlsafe_b64decode(
            tok + '=' * (-len(tok) % 4)).decode()
        if not hmac.compare_digest(sig, _share_sign(payload)):
            return None
        kind, ident, match_no, seat = payload.split('|')
        return kind, ident, int(match_no), int(seat)
    except Exception:
        return None


@app.get('/api/share')
def api_share(pid: str, sid: str = None, code: str = None,
              match_no: int = None):
    """Mint a read-only share token; ownership rules mirror /api/review."""
    pid = resolve_pid(pid)
    if code:
        lines = _log_lines_for(f'table:{code.upper()}')
        if not lines:
            raise HTTPException(404, 'no recorded deals for this table')
        want = match_no or max(l.get('match_no', 1) for l in lines)
        mlines = [l for l in lines if l.get('match_no', 1) == want]
        if not mlines:
            raise HTTPException(404, f'no deals for match {want}')
        seat = next((int(s) for s, p in (mlines[0].get('seat_pids') or {}).items()
                     if p == pid), None)
        if seat is None:
            raise HTTPException(403, 'you were not seated in this match')
        if (f'table:{code.upper()}', want) not in _finished_matches:
            raise HTTPException(409, 'sharing opens when the match ends')
        return {'token': _share_make('t', code.upper(), want, seat)}
    if sid:
        lines = [l for l in _log_lines_for(sid) if l.get('pid') == pid]
        if not lines:
            raise HTTPException(404, 'no recorded deals for this match')
        if (sid, 1) not in _finished_matches:
            raise HTTPException(409, 'sharing opens when the match ends')
        return {'token': _share_make('s', sid, 1, lines[0]['human_seat'])}
    raise HTTPException(400, 'sid or code required')


@app.get('/api/review')
def api_review(pid: str = None, sid: str = None, code: str = None,
               match_no: int = None, share: str = None):
    pid = resolve_pid(pid)
    if share:
        p = _share_parse(share)
        if p is None:
            raise HTTPException(403, 'invalid share link')
        kind, ident, want, seat = p
        if kind == 't':
            lines = [l for l in _log_lines_for(f'table:{ident}')
                     if l.get('match_no', 1) == want]
            key = (f'table:{ident}', want)
        else:
            lines = _log_lines_for(ident)
            key = (ident, 1)
        if not lines:
            raise HTTPException(404, 'this match is no longer available')
    elif code:
        if not pid:
            raise HTTPException(400, 'pid required')
        lines = _log_lines_for(f'table:{code.upper()}')
        if not lines:
            raise HTTPException(404, 'no recorded deals for this table')
        want = match_no or max(l.get('match_no', 1) for l in lines)
        lines = [l for l in lines if l.get('match_no', 1) == want]
        if not lines:
            raise HTTPException(404, f'no deals for match {want}')
        seat = next((int(s) for s, p in (lines[0].get('seat_pids') or {}).items()
                     if p == pid), None)
        if seat is None:
            raise HTTPException(403, 'you were not seated in this match')
        key = (f'table:{code.upper()}', want)
    elif sid:
        if not pid:
            raise HTTPException(400, 'pid required')
        lines = [l for l in _log_lines_for(sid) if l.get('pid') == pid]
        if not lines:
            raise HTTPException(404, 'no recorded deals for this match')
        seat = lines[0]['human_seat']
        key = (sid, 1)
    else:
        raise HTTPException(400, 'sid or code required')
    if key not in _finished_matches:
        raise HTTPException(409, 'the review opens when the match ends')
    cached = _review_cache.get(key)
    if cached is None or cached['n_deals'] != len(lines):
        cached = {'n_deals': len(lines),
                  'payload': compute_review(lines, -1)}
        _review_cache[key] = cached
        while len(_review_cache) > 20:
            _review_cache.pop(next(iter(_review_cache)))
    out = dict(cached['payload'])
    out['viewer_seat'] = seat
    # Codenames for human seats (per-request: pid-derived, retroactive -
    # matches logged before the codename system still resolve).
    seat_names = {}
    sp = lines[0].get('seat_pids') or {}
    for s2, p2 in sp.items():
        seat_names[str(s2)] = codename_of(p2)
    if not seat_names and lines[0].get('pid') is not None \
            and lines[0].get('human_seat') is not None:
        seat_names[str(lines[0]['human_seat'])] = codename_of(lines[0]['pid'])
    out['seat_names'] = seat_names
    # public profile handles per human seat (codename slugs - no
    # credentials anywhere in the public path)
    out['seat_players'] = {s2: _name_slug(n2)
                           for s2, n2 in seat_names.items()}
    return out


@app.get('/review')
def review_page():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'static', 'review.html'))


# ---------------------------------------------------------------------------
# Post-match insight: replay a finished match from the telemetry log and
# annotate every play-phase decision of ONE seat with the net's own choice.
# The AI's preference is a lens, not ground truth (rules #5 spirit).
# ---------------------------------------------------------------------------
def compute_insight(deal_lines, seat):
    # Replay collecting the seat's decision states, then ONE batched
    # forward (per-play forwards crawl when the box is busy).
    deal_lines = sorted(deal_lines, key=lambda d: d['deal_no'])
    menv = MatchEnv(seed=deal_lines[0]['seed'])
    obs_l, mask_l, meta = [], [], []
    for d in deal_lines:
        plays = 0
        played_deal = set()
        for s, card, ms in d['actions']:
            if menv.get_current_player() != s:
                raise HTTPException(500, 'replay desync in insight')
            passing = menv.is_passing()
            if not passing and s == seat:
                mask = np.zeros(52, dtype=bool)
                legal = []
                for a in menv.get_legal_actions():
                    if a != -1:
                        mask[a] = True
                        legal.append(a)
                obs = np.array(menv.observe(), dtype=np.float32)
                obs_l.append(obs)
                mask_l.append(mask)
                hand = set(np.flatnonzero(obs[:52] > 0).tolist())
                eq = _equiv_groups(hand, played_deal, legal)
                meta.append((d['deal_no'], plays // 4 + 1, card, len(legal), eq))
            if not passing:
                played_deal.add(card)
                plays += 1
            menv.step(card)
    n_dec, n_agree, dis = len(meta), 0, []
    if meta:
        obs_t = torch.from_numpy(np.stack(obs_l))
        mask_t = torch.from_numpy(np.stack(mask_l))
        with _net_lock, torch.no_grad():
            logits, _ = _net(obs_t, mask_t)
            probs = torch.softmax(logits, dim=1)
        ais = torch.argmax(logits, dim=1)
        for i, (deal_no, trick, card, n_legal, eq) in enumerate(meta):
            ai = int(ais[i])
            # Equivalent cards (connected string, visible info) count as
            # agreement - the net's preference inside a group is arbitrary.
            if ai == card or any(ai in g and card in g for g in eq):
                n_agree += 1
            elif n_legal > 1:
                dis.append({'deal': deal_no, 'trick': trick,
                            'you': card_name(card), 'ai': card_name(ai),
                            'p_you': round(float(probs[i, card]), 3),
                            'p_ai': round(float(probs[i, ai]), 3)})
    dis.sort(key=lambda x: x['p_ai'] - x['p_you'], reverse=True)
    return {'n_decisions': n_dec, 'n_agree': n_agree,
            'agree_pct': round(100 * n_agree / max(1, n_dec), 1),
            'disagreements': dis[:8],
            'note': ("Perilune's preference, not ground truth - the AI has "
                     "measured blind spots of its own.")}


# ---------------------------------------------------------------------------
# Server-assigned codenames (2026-08-08): every pid gets a permanent
# adjective-animal name, assigned ONCE on first contact and immutable by
# construction (no edit endpoint exists). Users never enter display text
# anywhere, so every public surface - tables, reviews, the eventual
# leaderboard - is moderation-free. Both lists are hand-curated with the
# cross-product in mind; the blocklist is cheap insurance on top.
# ---------------------------------------------------------------------------
CODENAME_ADJ = (
    'Laughing', 'Silent', 'Wandering', 'Iron', 'Lucky', 'Midnight', 'Sly',
    'Bold', 'Velvet', 'Thundering', 'Crescent', 'Waxing', 'Solar', 'Drifting',
    'Gilded', 'Umbral', 'Amber', 'Cobalt', 'Crimson', 'Ivory', 'Jade',
    'Quiet', 'Rambling', 'Dancing', 'Whistling', 'Humming', 'Gliding',
    'Soaring', 'Diving', 'Leaping', 'Prowling', 'Dozing', 'Dreaming',
    'Wistful', 'Merry', 'Dapper', 'Nimble', 'Stalwart', 'Gentle', 'Fearless',
    'Curious', 'Patient', 'Radiant', 'Dusky', 'Frosted', 'Blazing', 'Misty',
    'Starlit', 'Moonlit', 'Sunlit', 'Twilight', 'Auroral', 'Comet',
    'Meteoric', 'Orbital', 'Lunar', 'Stellar', 'Nebular', 'Zenith', 'Apogee',
    'Roaming', 'Marching', 'Sailing', 'Rowing', 'Trekking', 'Striding',
    'Galloping', 'Trotting', 'Pouncing', 'Perched', 'Burrowing', 'Nesting',
    'Clever', 'Wily', 'Canny', 'Shrewd', 'Stoic', 'Jolly', 'Sprightly',
    'Plucky', 'Daring', 'Vivid', 'Pale', 'Golden', 'Silver', 'Copper',
    'Bronze', 'Marble', 'Onyx', 'Opal', 'Coral', 'Indigo', 'Scarlet',
    'Emerald', 'Sapphire', 'Thoughtful', 'Whimsical', 'Serene', 'Spirited',
    'Steadfast', 'Vigilant', 'Wakeful', 'Winking', 'Grinning', 'Chuckling',
    'Humble', 'Noble', 'Regal', 'Rustic', 'Cosmic', 'Polar', 'Boreal',
    'Austral', 'Zephyr', 'Tidal', 'Rolling', 'Tumbling', 'Skipping',
)
CODENAME_ANIMAL = (
    'Turtle', 'Fox', 'Heron', 'Otter', 'Lynx', 'Sparrow', 'Badger', 'Orca',
    'Falcon', 'Owl', 'Crane', 'Ibis', 'Puffin', 'Petrel', 'Tern', 'Plover',
    'Marmot', 'Beaver', 'Hare', 'Ermine', 'Sable', 'Marten', 'Vole',
    'Hedgehog', 'Mole', 'Shrew', 'Dormouse', 'Squirrel', 'Chipmunk',
    'Raccoon', 'Panda', 'Koala', 'Wombat', 'Quokka', 'Kestrel', 'Osprey',
    'Condor', 'Albatross', 'Pelican', 'Cormorant', 'Kingfisher', 'Magpie',
    'Jackdaw', 'Raven', 'Rook', 'Starling', 'Swift', 'Swallow', 'Lark',
    'Finch', 'Wren', 'Robin', 'Thrush', 'Nightingale', 'Curlew', 'Sandpiper',
    'Gannet', 'Gull', 'Skua', 'Eider', 'Teal', 'Wigeon', 'Gadwall',
    'Pintail', 'Goldeneye', 'Merganser', 'Loon', 'Grebe', 'Bittern',
    'Egret', 'Stork', 'Spoonbill', 'Flamingo', 'Avocet', 'Stilt', 'Godwit',
    'Whimbrel', 'Turnstone', 'Dunlin', 'Knot', 'Sanderling', 'Dotterel',
    'Lapwing', 'Seal', 'Walrus', 'Narwhal', 'Beluga', 'Dolphin', 'Porpoise',
    'Manatee', 'Dugong', 'Gazelle', 'Ibex', 'Chamois', 'Reindeer', 'Caribou',
    'Elk', 'Moose', 'Tapir', 'Okapi', 'Oryx', 'Kudu', 'Eland', 'Impala',
    'Springbok', 'Meerkat', 'Mongoose', 'Genet', 'Civet', 'Serval',
    'Caracal', 'Ocelot', 'Margay', 'Pangolin', 'Armadillo', 'Sloth',
    'Tamarin', 'Marmoset', 'Capybara', 'Agouti', 'Chinchilla', 'Degu',
    'Jerboa', 'Pika', 'Lemming', 'Muskrat', 'Nutria', 'Coypu', 'Axolotl',
    'Newt', 'Gecko', 'Skink', 'Tortoise', 'Terrapin',
)
# Stems precise enough not to false-positive on curated words (a bare
# 'nig' would ban Midnight and Nightingale; the audit in the commit
# checks the full cross-product on every list change).
_NAME_BLOCK = ('ass', 'sex', 'tit', 'cum', 'fag', 'nigg', 'rape', 'nazi')
NAMES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'player_names.jsonl')
_names = {}
_names_used = set()
_hash_era = set()   # canonical ids that are HASHES (never credentials)
_names_lock = threading.Lock()
try:
    with open(NAMES_PATH, encoding='utf-8') as _f:
        for _line in _f:
            try:
                _d = json.loads(_line)
                if _d.get('he'):
                    _hash_era.add(_d['pid'])
                _names[_d['pid']] = _d['name']
                _names_used.add(_d['name'])
            except (ValueError, KeyError):
                continue
except FileNotFoundError:
    pass

# Public handles: the codename itself, slugified - unique by construction,
# permanent, human-readable. /player/<slug> is the profile URL; the key
# never appears anywhere in the public path.
def _name_slug(n):
    return n.lower().replace(' ', '-')


_slug2pid = {_name_slug(_n): _p for _p, _n in _names.items()}


def codename_of(pid, hash_era=False):
    pid = (pid or '')[:64]
    if not pid:
        return 'Nameless Drifter'
    with _names_lock:
        if pid in _names:
            return _names[pid]
        if hash_era:
            _hash_era.add(pid)
        rng = secrets.SystemRandom()
        name = None
        for _ in range(500):
            cand = f'{rng.choice(CODENAME_ADJ)} {rng.choice(CODENAME_ANIMAL)}'
            if cand in _names_used:
                continue
            low = cand.replace(' ', '').lower()
            if any(b in low for b in _NAME_BLOCK):
                continue
            name = cand
            break
        if name is None:   # pool thin: numbered combo keeps the flavor
            for _ in range(500):
                cand = (f'{rng.choice(CODENAME_ADJ)} '
                        f'{rng.choice(CODENAME_ANIMAL)} {rng.randrange(10, 100)}')
                if cand not in _names_used:
                    name = cand
                    break
        if name is None:   # unreachable at this scale; never fail
            name = f'Wanderer {secrets.token_hex(3)}'
        _names[pid] = name
        _names_used.add(name)
        _slug2pid[_name_slug(name)] = pid
        with open(NAMES_PATH, 'a', encoding='utf-8') as f:
            row = {'pid': pid, 'name': name, 'ts': int(time.time())}
            if hash_era:
                row['he'] = 1
            f.write(json.dumps(row) + '\n')
        return name


# ---------------------------------------------------------------------------
# Identity keys (2026-08-08): the pid IS the bearer key - never displayed,
# never in page URLs, stored only in the player's browser (and their own
# key-file backup). Identities are minted LAZILY: only /api/identity/new
# creates one (the client calls it at the first action that needs an
# identity), so visiting the site never creates a dead account. Restore =
# presenting an existing key; /api/identity/check never mints. Legacy
# client-generated pids remain valid keys (they are the identity in every
# log), so existing players migrate transparently.
# ---------------------------------------------------------------------------
@app.post('/api/identity/new')
def identity_new():
    for _ in range(100):
        key = secrets.token_hex(16)
        canon = _kh(key)
        with _names_lock:
            taken = canon in _names
        if not taken:
            break
    # The canonical id is the key's HASH: the plaintext key never touches
    # a server file. codename_of registers assign-once; a 128-bit
    # collision between check and assignment is not a real event.
    return {'key': key, 'name': codename_of(canon, hash_era=True)}


# Keys at rest (2026-08-08, public-launch prep): the server stores only
# SHA-256 fingerprints of credentials. For hash-era identities the
# canonical id IS the key's hash - the plaintext key exists only in the
# player's browser and key-file, so a leaked server file impersonates
# nobody. Canonical ids are IDENTIFIERS, never credentials: presenting a
# hash-era canonical id is rejected outright (only the preimage - the
# key - authenticates). Legacy identities (pre-hash raw pids) keep
# working as-is; their exposure closes on rotation/upgrade, after which
# their logged pid is just a revoked id. Rotation maps a new key's hash
# to the same canonical id and revokes the old credential.
KEYS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'player_keys.jsonl')


def _kh(k):
    return hashlib.sha256(k.encode()).hexdigest()


_cred2canon = {}   # sha256(current key) -> canonical id (rotated ids)
_canon2cred = {}   # canonical id -> sha256(current key)
_revoked = set()   # raw legacy creds AND cred-hashes rotated away
try:
    with open(KEYS_PATH, encoding='utf-8') as _f:
        for _line in _f:
            try:
                _d = json.loads(_line)
                # v1 events stored the raw replacement key; hash on load
                _nh = _d.get('kh') or _kh(_d['key'])
                for _r in _d.get('rv', ()):
                    _revoked.add(_r)
                    _cred2canon.pop(_r, None)
                _prevh = _canon2cred.get(_d['canon'])
                if _prevh:
                    _revoked.add(_prevh)
                    _cred2canon.pop(_prevh, None)
                else:
                    # first rotation: the canonical itself was the cred
                    _revoked.add(_d['canon'])
                _cred2canon[_nh] = _d['canon']
                _canon2cred[_d['canon']] = _nh
                _revoked.discard(_nh)
            except (ValueError, KeyError):
                continue
except FileNotFoundError:
    pass


def resolve_pid(key):
    """Canonical id for a presented CREDENTIAL. Revoked keys fail loudly
    (a stale device explains itself); hash-era canonical ids are refused
    - identifiers in server files must never authenticate."""
    k = (key or '')[:64]
    if not k:
        return k
    with _names_lock:
        h = _kh(k)
        if k in _revoked or h in _revoked:
            raise HTTPException(
                404, 'this key was rotated - use your replacement key')
        c = _cred2canon.get(h)
        if c:
            return c            # rotated identity, current key
        if h in _names:
            return h            # hash-era identity, original key
        if k in _names and k in _hash_era:
            raise HTTPException(403, 'invalid key')   # id, not a credential
        return k                # legacy credential, or unknown (mints later)


class RotateBody(BaseModel):
    key: str


@app.post('/api/identity/rotate')
def identity_rotate(body: RotateBody):
    with _names_lock:
        k = (body.key or '')[:64]
        h = _kh(k)
        if k in _revoked or h in _revoked:
            raise HTTPException(404, 'this key was already rotated')
        canon = _cred2canon.get(h) or (h if h in _names else k)
        if canon not in _names or (k in _names and k in _hash_era):
            raise HTTPException(404, 'unknown key')
        nk = secrets.token_hex(16)
        nh = _kh(nk)
        rv = [x for x in (k, h, _canon2cred.get(canon)) if x]
        for x in rv:
            _revoked.add(x)
            _cred2canon.pop(x, None)
        _cred2canon[nh] = canon
        _canon2cred[canon] = nh
        _revoked.discard(nh)
        with open(KEYS_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'canon': canon, 'kh': nh, 'rv': rv,
                                'ts': int(time.time())}) + '\n')
        return {'key': nk, 'name': _names[canon]}


@app.get('/api/identity/check')
def identity_check(key: str):
    """Non-minting lookup: does this key name an existing identity?"""
    canon = resolve_pid(key)
    with _names_lock:
        name = _names.get(canon)
    if name is None:
        raise HTTPException(404, 'unknown key')
    return {'ok': True, 'name': name}


@app.get('/api/name')
def api_name(pid: str):
    """Non-minting (identity creation is /api/identity/new only)."""
    canon = resolve_pid(pid)
    with _names_lock:
        name = _names.get(canon)
    if name is None:
        raise HTTPException(404, 'unknown player')
    return {'name': name}


# ---------------------------------------------------------------------------
# Progress page: per-match skill stats for one player, computed lazily by
# replay + one batched forward, cached per match (matches never change
# once finished). Agreement is habit-authority (the net's own preference,
# not ground truth); readability is belief-authority - both labeled so on
# the page.
# ---------------------------------------------------------------------------
_progress_cache = {}


def compute_match_stats(deal_lines, seat):
    deal_lines = sorted(deal_lines, key=lambda d: d['deal_no'])
    seat_types = deal_lines[0].get('seats') or {}
    ai_seats = [s for s in range(4)
                if seat_types.get(str(s), 'ai') != 'human' and s != seat]
    menv = MatchEnv(seed=deal_lines[0]['seed'])
    obs_l, mask_l = [], []
    agree_meta = []   # (row, category, card, n_legal, eq) for the seat's picks
    bel_meta = []     # (row, observer, play_idx) belief rows, all seats
    play_hands = []   # per play state: {seat: frozenset(cards)} for tracked
    play_deal = []    # per play state: (deal_idx, play_in_deal)
    tracked = sorted({seat, *ai_seats})
    moons_shot = moons_conceded = 0
    n_deals_ct = deals_zero = deals_harsh = 0
    for di, d in enumerate(deal_lines):
        plays = 0
        played_deal = set()
        trick = []
        for s, card, ms in d['actions']:
            if menv.get_current_player() != s:
                raise ValueError('replay desync in progress stats')
            passing = menv.is_passing()
            if passing:
                if s == seat:
                    mask = np.zeros(52, dtype=bool)
                    legal = [a for a in menv.get_legal_actions() if a != -1]
                    for a in legal:
                        mask[a] = True
                    obs = np.array(menv.observe(), dtype=np.float32)
                    hand = set(np.flatnonzero(obs[:52] > 0).tolist())
                    agree_meta.append((len(obs_l), 'pass', card, len(legal),
                                       _equiv_groups(hand, set(), legal)))
                    obs_l.append(obs)
                    mask_l.append(mask)
            else:
                mask = np.zeros(52, dtype=bool)
                legal = [a for a in menv.get_legal_actions() if a != -1]
                for a in legal:
                    mask[a] = True
                # belief observers for readability: every tracked seat's view
                pi = len(play_hands)
                hands_now = {}
                for o in range(4):
                    ob = np.array(menv.observe_for(o), dtype=np.float32)
                    if o in tracked:
                        hands_now[o] = frozenset(
                            np.flatnonzero(ob[:52] > 0).tolist())
                    bel_meta.append((len(obs_l), o, pi))
                    obs_l.append(ob)
                    mask_l.append(mask)
                play_hands.append(hands_now)
                play_deal.append((di, plays))
                if s == seat:
                    cat = ('lead' if not trick
                           else 'follow' if card // 13 == trick[0] // 13
                           else 'discard')
                    obs = np.array(menv.observe(), dtype=np.float32)
                    hand = set(np.flatnonzero(obs[:52] > 0).tolist())
                    agree_meta.append((len(obs_l), cat, card, len(legal),
                                       _equiv_groups(hand, played_deal, legal)))
                    obs_l.append(obs)
                    mask_l.append(mask)
                trick.append(card)
                if len(trick) == 4:
                    trick = []
                played_deal.add(card)
                plays += 1
            menv.step(card)
        rs = d['round_scores']
        if sum(rs) == 78:
            shooter = int(np.argmin(rs))
            if shooter == seat:
                moons_shot += 1
            else:
                moons_conceded += 1
        n_deals_ct += 1
        if rs[seat] == 0:
            deals_zero += 1
        if rs[seat] >= 25:   # 25 = maximal non-moon damage, 26 = ate a moon
            deals_harsh += 1
    # one batched forward for everything
    agree = {c: [0, 0] for c in ('pass', 'lead', 'follow', 'discard')}
    read_t6 = {t: [] for t in tracked}
    if obs_l:
        obs_t = torch.from_numpy(np.stack(obs_l))
        mask_t = torch.from_numpy(np.stack(mask_l))
        lg_l, bel_l = [], []
        with _net_lock, torch.no_grad():
            for i in range(0, len(obs_l), 512):
                lg, _, bl = _net.forward_all(obs_t[i:i + 512],
                                             mask_t[i:i + 512])
                lg_l.append(lg)
                bel_l.append(torch.sigmoid(bl))
        logits = torch.cat(lg_l)
        beliefs = torch.cat(bel_l)
        ais = torch.argmax(logits, dim=1)
        for row, cat, card, n_legal, eq in agree_meta:
            if n_legal < 2:
                continue
            ai = int(ais[row])
            agree[cat][1] += 1
            if ai == card or any(ai in g and card in g for g in eq):
                agree[cat][0] += 1
        # readability: per play, each tracked target's remaining hand scored
        # by its three observers' normalized located-confidence; snapshot the
        # value at the last play of trick 6 per deal, averaged over deals
        obs_rows = {}   # (play_idx, observer) -> forward row
        for row, o, pi in bel_meta:
            obs_rows[(pi, o)] = row
        for pi, hands_now in enumerate(play_hands):
            di, plays = play_deal[pi]
            if plays != 23:
                continue
            for t, hand in hands_now.items():
                if not hand:
                    continue
                tot_conf = n_conf = 0
                for o in range(4):
                    if o == t:
                        continue
                    b = beliefs[obs_rows[(pi, o)]]
                    k = (t - o) % 4 - 1
                    for c in hand:
                        ps = [float(b[r * 52 + c]) for r in range(3)]
                        z = sum(ps) or 1.0
                        tot_conf += ps[k] / z
                        n_conf += 1
                if n_conf:
                    read_t6[t].append(tot_conf / n_conf)
    ai_vals = [v for t in ai_seats for v in read_t6.get(t, ())]
    my_vals = read_t6.get(seat, ())
    return {
        'agree': agree,
        'agree_pct': round(100 * sum(v[0] for v in agree.values())
                           / max(1, sum(v[1] for v in agree.values())), 1),
        'read_t6': round(100 * sum(my_vals) / len(my_vals), 1) if my_vals else None,
        'read_t6_ai': round(100 * sum(ai_vals) / len(ai_vals), 1) if ai_vals else None,
        'moons_shot': moons_shot, 'moons_conceded': moons_conceded,
        'n_deals': n_deals_ct, 'deals_zero': deals_zero,
        'deals_harsh': deals_harsh,
    }


@app.get('/api/progress')
def api_progress(pid: str = None, player: str = None, limit: int = 60):
    """Own view (pid = the key) or PUBLIC view (player = codename slug).
    Public rows drop the sid and carry a minted share token per match -
    profile visitors open reviews through the exact read-only path share
    links use; no new access model exists."""
    public = False
    pub_name = None
    if player:
        with _names_lock:
            canon = _slug2pid.get((player or '').strip().lower())
            pub_name = _names.get(canon) if canon else None
        if canon is None or pub_name is None:
            raise HTTPException(404, 'unknown player')
        pid = canon
        public = True
        limit = min(limit, 40)   # bound anonymous compute per request
    else:
        pid = resolve_pid(pid)
    hist = list(_idx_history.get(pid, ()))[-max(1, min(100, limit)):]
    out = []
    for h in hist:
        if h['mode'] == 'table':
            key = (f"table:{h['code']}", h['match_no'])
            lines = [l for l in _log_lines_for(key[0])
                     if l.get('match_no', 1) == h['match_no']]
        else:
            key = (h['sid'], 1)
            lines = _log_lines_for(h['sid'])
        st = _progress_cache.get(key)
        if st is None:
            try:
                st = compute_match_stats(lines, h['seat']) if lines else {}
            except Exception:
                st = {}
            _progress_cache[key] = st
            while len(_progress_cache) > 200:
                _progress_cache.pop(next(iter(_progress_cache)))
        row = {**h, **st}
        if public:
            row.pop('sid', None)
            row['share'] = (_share_make('t', h['code'].upper(),
                                        h['match_no'], h['seat'])
                            if h['mode'] == 'table'
                            else _share_make('s', h['sid'], 1, h['seat']))
        out.append(row)
    return {'matches': out, 'public': public, 'name': pub_name}


@app.get('/progress')
def progress_page():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'static', 'progress.html'))


@app.get('/account')
def account_page():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'static', 'account.html'))


@app.get('/player/{slug}')
def player_page(slug: str):
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'static', 'progress.html'))


@app.get('/api/leaderboard')
def api_leaderboard(era: str = None):
    """Current model-era board by default; archived eras by md5. Every
    row links its winning match via a minted share token - every score
    is verifiable by inspection."""
    cur_era = MODEL_MD5[:12]
    e = (era or cur_era)[:12]
    with _log_lock:
        entries = list(_lb.get(e, {}).values())
        eras = [{'era': k, 'n': len(v),
                 'latest': max(r['ts'] for r in v.values())}
                for k, v in _lb.items() if v]
    entries.sort(key=lambda r: (r['score'], r['ts']))
    rows = []
    for i, r in enumerate(entries[:100]):
        nm = codename_of(r['canon'])
        rows.append({'rank': i + 1, 'name': nm, 'slug': _name_slug(nm),
                     'score': r['score'], 'deals': r['deals'], 'ts': r['ts'],
                     'share': _share_make('s', r['sid'], 1, r['seat'])})
    eras.sort(key=lambda x: x['latest'], reverse=True)
    return {'era': e, 'current': e == cur_era, 'current_era': cur_era,
            'eras': eras, 'rows': rows}


@app.get('/leaderboard')
def leaderboard_page():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'static', 'leaderboard.html'))


def _log_lines_for(sid):
    """Deal lines for one sid via the index: seek straight to its lines
    instead of scanning the whole log."""
    out = []
    with _log_lock:
        offs = list(_idx_deals.get(sid, ()))
        try:
            with open(LOG_PATH, 'rb') as f:
                for off in offs:
                    f.seek(off)
                    try:
                        out.append(json.loads(f.readline().decode()))
                    except ValueError:
                        continue
        except FileNotFoundError:
            pass
    return out


@app.get('/api/insight/{sid}')
def solo_insight(sid: str, pid: str):
    pid = resolve_pid(pid)
    if (sid, 1) not in _finished_matches:
        raise HTTPException(409, 'insights open when the match ends')
    lines = [l for l in _log_lines_for(sid) if l.get('pid') == pid]
    if not lines:
        raise HTTPException(404, 'no recorded deals for this match')
    return compute_insight(lines, lines[0]['human_seat'])


@app.get('/api/table/insight/{code}')
def table_insight(code: str, pid: str):
    pid = resolve_pid(pid)
    lines = _log_lines_for(f'table:{code.upper()}')
    if not lines:
        raise HTTPException(404, 'no recorded deals for this table')
    latest = max(l.get('match_no', 1) for l in lines)
    if (f'table:{code.upper()}', latest) not in _finished_matches:
        raise HTTPException(409, 'insights open when the match ends')
    lines = [l for l in lines if l.get('match_no', 1) == latest]
    seat = next((int(s) for s, p in (lines[0].get('seat_pids') or {}).items()
                 if p == pid), None)
    if seat is None:
        raise HTTPException(403, 'you were not seated in this match')
    return compute_insight(lines, seat)


@app.get('/api/history')
def api_history(pid: str, limit: int = 12):
    pid = resolve_pid(pid)
    """The pid's completed matches, newest first (menu 'Recent matches').
    Served straight from the in-memory index - no log scan."""
    with _log_lock:
        out = list(_idx_history.get(pid, ()))
    out.sort(key=lambda m: m['ts'], reverse=True)
    return {'matches': out[:limit]}


@app.get('/about')
def about():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'static', 'about.html'))


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
    local = (cfg.DEV_MODE == 'localhost'
             and request.headers.get('cf-connecting-ip') is None
             and request.client is not None
             and request.client.host in ('127.0.0.1', '::1'))
    if not local:
        html = html.replace('const DEV_CONTROLS = true;',
                            'const DEV_CONTROLS = false;', 1)
    return Response(content=html, media_type='text/html')


@app.post('/api/new')
def new_session(body: NewBody | None = None, request: Request = None):
    pid = (resolve_pid(body.pid) if body and body.pid else None)
    ua = request.headers.get('user-agent', '') if request else ''
    s = Session(pid=pid, ua=ua, tier=(body.tier if body else None) or 'full')
    with _sessions_lock:
        _sessions[s.sid] = s
        # Drop oldest sessions past a sane cap
        while len(_sessions) > cfg.SESSION_CAP:
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


# A sid alone is NOT a credential: solo review URLs carry it in the
# address bar (visible to anyone watching a stream), so pid-bound
# sessions require the pid on state reads and plays - otherwise a
# spectator could read the hand or inject moves into a live session.
def _own_session(s, pid):
    if s.pid and (not pid or resolve_pid(pid) != s.pid):
        raise HTTPException(403, 'not your session')


@app.get('/api/state/{sid}')
def get_state(sid: str, pid: str = None):
    s = _get(sid)
    _own_session(s, pid)
    with s.lock:
        return s.state()


class TableNewBody(BaseModel):
    pid: str
    name: str | None = None
    tier: str | None = None


class TableJoinBody(BaseModel):
    code: str
    pid: str
    name: str | None = None
    timer_s: int | None = None   # host's turn-timer choice, sent with start


class TableActBody(BaseModel):
    pid: str
    card: int | None = None
    cards: list[int] | None = None
    cursor: int = 0


# (body.name is still accepted for old clients but IGNORED everywhere:
# display names are server-assigned codenames, immutable by construction.)
@app.post('/api/table/new')
def table_new(body: TableNewBody):
    canon = resolve_pid(body.pid)
    t = Table(canon, codename_of(canon),
              tier=body.tier or 'full')
    with _tables_lock:
        while t.code in _tables:
            t.code = ''.join(secrets.choice(CODE_ALPHABET)
                             for _ in range(cfg.CODE_LEN))
        # Drop stale tables past a sane cap (oldest first).
        while len(_tables) > cfg.TABLE_CAP:
            _tables.pop(next(iter(_tables)))
        _tables[t.code] = t
    return t.view(canon)


@app.post('/api/table/join')
def table_join(body: TableJoinBody):
    t = _get_table(body.code)
    with t.lock:
        pid = resolve_pid(body.pid)
        if any(p['pid'] == pid for p in t.lobby) or pid in t.seat_of:
            return t.view(pid)          # idempotent rejoin
        if t.state != 'lobby':
            raise HTTPException(409, 'match already in progress - ask the '
                                     'host to close and start a new table')
        if len(t.lobby) >= 4:
            raise HTTPException(409, 'table is full (4 players max)')
        t.lobby.append({'pid': pid, 'name': codename_of(pid)})
        return t.view(pid)


@app.post('/api/table/start')
def table_start(body: TableJoinBody):
    t = _get_table(body.code)
    body.pid = resolve_pid(body.pid)
    with t.lock:
        if body.pid != t.host_pid:
            raise HTTPException(403, 'only the host can start')
        if t.state == 'lobby':
            if body.timer_s in (0, 30, 60, 90, 120):
                t.timer_s = body.timer_s
            t.start()
        return t.view(body.pid)


@app.get('/api/table/state/{code}')
def table_state(code: str, pid: str, cursor: int = 0):
    t = _get_table(code)
    pid = resolve_pid(pid)
    with t.lock:
        t.check_timeout()
        return t.view(pid, cursor)


@app.post('/api/table/rematch')
def table_rematch(body: TableJoinBody):
    """Host-only, finished matches only: new match, same table and seats."""
    t = _get_table(body.code)
    body.pid = resolve_pid(body.pid)
    with t.lock:
        if body.pid != t.host_pid:
            raise HTTPException(403, 'only the host can start a rematch')
        if t.state != 'playing' or not t.finished:
            raise HTTPException(409, 'rematch is only available after a match ends')
        t.rematch()
        return t.view(body.pid, 0)


@app.post('/api/table/close')
def table_close(body: TableJoinBody):
    """Host-only: delete the table for everyone (end-game screen button).
    Other players' next poll gets 404 and returns to the menu cleanly."""
    t = _get_table(body.code)
    body.pid = resolve_pid(body.pid)
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
        pid = resolve_pid(body.pid)
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
        time.sleep(cfg.REAPER_INTERVAL_S)
        now = time.time()
        with _tables_lock:
            items = list(_tables.items())
        for code, t in items:
            with t.lock:
                pids = t.human_pids()
                drop = (not pids or
                        all(p in t.departed
                            or now - t.last_seen.get(p, t.created) > cfg.STALE_S
                            for p in pids))
            if drop:
                with _tables_lock:
                    _tables.pop(code, None)


threading.Thread(target=_reaper, daemon=True).start()


@app.post('/api/table/pass/{code}')
def table_pass(code: str, body: TableActBody):
    t = _get_table(code)
    body.pid = resolve_pid(body.pid)
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
    body.pid = resolve_pid(body.pid)
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
    _own_session(s, body.pid)
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
