"""Regression suite for the web app. Run before every deploy:

    python hearts_web/test_site.py            # all groups
    python hearts_web/test_site.py analysis   # one group (substring match)

No pytest dependency (it is not installed here) and no network: the API
tests drive the app in-process through FastAPI's TestClient. Exit code 0
= everything passed, 1 = at least one failure.

TWO THINGS THIS FILE MUST DO BEFORE IMPORTING THE SERVER, both learned
the hard way:
  * point HEARTS_LOG_PATH at a temp file - a scratch server writing the
    REAL match log put a test identity on the public leaderboard
    (2026-08-12);
  * point HEARTS_FEEDBACK_PATH at a temp file for the same reason;
  * stub the off-host backup starter - importing the server otherwise
    spawns a thread that uploads the live data files to R2.

Each test states the invariant it protects and, where it came from a
real incident, the date. A failure message should tell you what broke
and why it matters, not just which assert tripped.
"""
import json
import os
import re
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, 'hearts_web')
STATIC = os.path.join(WEB, 'static')
sys.path.insert(0, ROOT)

# --- isolation valves, BEFORE the server import ---------------------------
_tmp_log = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
_tmp_log.close()
os.environ['HEARTS_LOG_PATH'] = _tmp_log.name
_tmp_fb = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
_tmp_fb.close()
os.environ['HEARTS_FEEDBACK_PATH'] = _tmp_fb.name
# Client-search store gets its own valve for the same reason the match log
# does: test matches must never land in real player data.
_tmp_cs = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
_tmp_cs.close()
os.environ['HEARTS_CS_LOG_PATH'] = _tmp_cs.name
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')       # CPU: tests are tiny

import hearts_web.backup_sync as _bs                     # noqa: E402
_bs.start_from_config = lambda cfg: None                 # no R2 uploads

from fastapi.testclient import TestClient                # noqa: E402
from hearts_web import server                            # noqa: E402

# TestClient's default client host is the literal 'testclient',
# which the loopback guard rightly refuses - give it a real one
client = TestClient(server.app, client=('127.0.0.1', 51234))
LOCAL = {}                       # TestClient looks like 127.0.0.1 = admin OK
TUNNEL = {'CF-Connecting-IP': '203.0.113.7'}   # looks like public traffic

TESTS = []


def test(group):
    def deco(fn):
        fn.group = group
        TESTS.append(fn)
        return fn
    return deco


def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def _feedback_rows():
    """Rows currently in the live feedback file (may be absent)."""
    try:
        with open(server.FEEDBACK_PATH, encoding='utf-8') as f:
            return [json.loads(ln) for ln in f if ln.strip()]
    except FileNotFoundError:
        return []


# ===========================================================================
# analysis - the maths the review/practice panels report
# ===========================================================================

@test('analysis')
def equivalence_never_bridges_a_card_on_the_table():
    """A card ON THE TABLE is a live threshold, not a spent card.

    Reported 2026-08-13: with the 5C led and 4C..AC in hand, all eight
    merged into one bar because the 5C was in `played`. The 4C ducks the
    trick and the 6C beats it - and a group SHARES one search estimate,
    so a wrong bridge corrupts the analysis, not just the display.
    """
    C = lambda rank: rank - 2                      # 2C=0 .. AC=12
    hand = {C(4), C(6), C(8), C(9), C(10), C(12), C(13), C(14)}
    played = {C(5), C(7), C(11)}                   # 5C led; 7C, JC spent
    legal = sorted(hand)

    groups = server._equiv_groups(hand, played, legal, table={C(5)})
    flat = [c for g in groups for c in g]
    assert C(4) not in flat, (
        '4C grouped with cards that BEAT the led 5C - the table card was '
        'treated as a bridge again')
    assert any(set(g) == {C(6), C(8), C(9), C(10), C(12), C(13), C(14)}
               for g in groups), (
        'the genuinely equivalent 6C..AC run should still group (7C and '
        'JC are spent, so nobody can hold a card between them)')


@test('analysis')
def equivalence_still_bridges_a_spent_card():
    """The same 5C, played in an EARLIER trick, IS a valid bridge."""
    C = lambda rank: rank - 2
    hand = {C(4), C(6), C(8), C(9), C(10), C(12), C(13), C(14)}
    played = {C(5), C(7), C(11)}
    groups = server._equiv_groups(hand, played, sorted(hand), table=())
    assert len(groups) == 1 and len(groups[0]) == 8, (
        'grouping was lost for cards nobody can hold a separator for - '
        'the fix over-corrected and the optimisation is gone')


@test('analysis')
def queen_of_spades_is_never_equivalent():
    """13 points vs 0 for her neighbours: she must never share a bar."""
    QS = server._QS
    hand = {QS - 1, QS, QS + 1}
    groups = server._equiv_groups(hand, set(), sorted(hand), table=())
    assert all(QS not in g for g in groups), 'the QS joined an equivalence group'


@test('analysis')
def table_cards_come_from_the_observation_block():
    """_table_cards must read obs[52:104] - the trick in progress."""
    import numpy as np
    obs = np.zeros(882, dtype=np.float32)
    obs[0] = 1.0            # a card in hand
    obs[52 + 17] = 1.0      # card 17 on the table
    obs[104 + 30] = 1.0     # card 30 in the completed history
    assert server._table_cards(obs) == {17}, (
        '_table_cards read the wrong observation block - hand or history '
        'leaked into the trick-in-progress set')


# ===========================================================================
# pages + api - does the site actually serve
# ===========================================================================

@test('pages')
def every_public_page_serves():
    for path in ['/', '/about', '/how', '/support', '/leaderboard',
                 '/profile', '/account']:
        r = client.get(path)
        assert r.status_code == 200, f'{path} returned {r.status_code}'
        assert len(r.content) > 500, f'{path} served a suspiciously tiny body'


@test('pages')
def leaderboard_api_shape():
    r = client.get('/api/leaderboard')
    assert r.status_code == 200
    d = r.json()
    for key in ('era', 'current_era', 'rows'):
        assert key in d, f'/api/leaderboard lost its "{key}" field'
    assert isinstance(d['rows'], list)


@test('pages')
def unknown_route_404s():
    assert client.get('/no-such-page-xyz').status_code == 404


# ===========================================================================
# gameplay - the engine/server integration
# ===========================================================================

@test('gameplay')
def a_solo_match_deals_and_accepts_a_play():
    r = client.post('/api/new', json={})
    assert r.status_code == 200, f'/api/new failed: {r.status_code}'
    sid = r.json()['sid']

    st = client.get(f'/api/state/{sid}').json()
    assert len(st['hand']) == 13, f"dealt {len(st['hand'])} cards, expected 13"
    assert st['legal'], 'no legal actions offered at the start of a match'

    # passing picks and plays share /api/play; during the pass phase the
    # first three calls are the three cards handed across
    before = len(st['hand'])
    r = client.post(f'/api/play/{sid}', json={'card': st['legal'][0]})
    assert r.status_code == 200, f'play rejected: {r.status_code} {r.text}'
    after = client.get(f'/api/state/{sid}').json()
    assert len(after['hand']) < before or after.get('passed_so_far'), (
        'submitting a legal card neither removed it from the hand nor '
        'registered as a pass pick')


@test('gameplay')
def a_fresh_deal_reports_hearts_unbroken():
    """2026-08-15: the hearts-broken badge was derived client-side from
    watched plays, so a client that never watched the heart land - a
    refresh, a resumed session, a spectator - showed it wrong. The state
    payload carries the flag now, and /api/state is exactly what a
    refresh fetches, so the key must be present and must start False."""
    sid = client.post('/api/new', json={'practice': True}).json()['sid']
    st = client.get(f'/api/state/{sid}').json()
    assert 'hearts_broken' in st, (
        '/api/state dropped hearts_broken - the client silently falls back '
        'to watched plays, and a mid-deal refresh shows a stale badge')
    assert st['hearts_broken'] is False, 'a fresh deal starts unbroken'


# ---- client-search play ---------------------------------------------------
# The AI's move is chosen by the visitor's browser, so the visitor's machine
# necessarily holds the AI's cards. These guard the two things that keeps
# survivable: the disclosure stays confined to client-search sessions, and
# these matches never touch the trusted store.

def _cs_new(want_awaiting=False):
    """A client-search session. With want_awaiting, drive past the pass
    phase (which the server plays) until an AI seat is on turn for a PLAY -
    a test that quietly returns early when the human is up is a test that
    covers nothing on those deals."""
    for _ in range(12):
        r = client.post('/api/new', json={'client_search': True})
        assert r.status_code == 200, f'/api/new client_search failed: {r.text}'
        st = r.json()
        if not want_awaiting or st.get('awaiting'):
            return st
        # the human may be first to act; play until an AI seat is awaited
        for _ in range(6):
            if not (st.get('your_turn') and st.get('legal')):
                break
            st = client.post(f"/api/play/{st['sid']}",
                             json={'card': st['legal'][0]}).json()
            if st.get('awaiting'):
                return st
    raise AssertionError('12 client-search deals and never an AI seat on turn')


@test('gameplay')
def an_ordinary_session_never_exposes_another_seats_hand():
    """The highest-severity regression in this feature: the client-search
    disclosure bleeding into normal play. A normal state payload carries
    only the human's own hand, and /api/cs/position must refuse outright."""
    st = client.post('/api/new', json={}).json()
    sid = st['sid']
    for key in ('start_hands', 'hands', 'ai_hands'):
        assert key not in st, f'normal state payload exposed {key!r}'
    assert st.get('client_search') is False
    r = client.get(f'/api/cs/position/{sid}')
    assert r.status_code == 409, (
        f'/api/cs/position answered a NORMAL session ({r.status_code}) - '
        'the AI hand disclosure is reachable outside client-search play')


@test('gameplay')
def client_search_waits_for_the_browser_instead_of_playing_the_ai():
    """The whole mode rests on run_ai_turns() stopping. If the server plays
    the AI anyway, the client's search result arrives for a position that
    has already moved on."""
    st = _cs_new(want_awaiting=True)
    assert st['client_search'] is True
    assert st['search_profile'] == {'play_k': 64, 'pass_k': 24}, (
        f"profile drifted from the teacher: {st['search_profile']}")
    aw = st['awaiting']
    assert aw['seat'] != st['your_seat'], 'awaiting the human seat'
    assert (aw['phase'], aw['k']) in (('play', 64), ('pass', 24)), (
        f'awaited {aw["phase"]} at K={aw["k"]}; the teacher is play 64 / '
        'pass 24')
    if aw['phase'] == 'pass':
        assert aw['cards'] == 3, (
            'a pass is scored as whole 3-card combos, so the client must be '
            'told to send three')
    pos = client.get(f"/api/cs/position/{st['sid']}").json()
    # the WHOLE match so far, one entry per deal: pass direction is
    # deal_count % 4 and the engine counts deals by replaying them, so a
    # single-deal payload passes the wrong way from deal 2 on
    n = pos['deal_idx'] + 1
    assert len(pos['deals']) == n and len(pos['start_hands']) == n, (
        f"position sent {len(pos['deals'])} deals and "
        f"{len(pos['start_hands'])} hand sets for deal_idx {pos['deal_idx']}")
    live = pos['start_hands'][pos['deal_idx']]
    assert len(live) == 4 and sum(len(h) for h in live) == 52, (
        'the live deal\'s start hands are not a full 52-card deal')
    assert pos['action_idx'] == len(pos['deals'][pos['deal_idx']])
    assert pos['legal'], 'no legal actions offered for the AI seat'


@test('gameplay')
def client_search_rejects_moves_it_should_not_accept():
    st = _cs_new(want_awaiting=True)
    sid = st['sid']
    aw = st['awaiting']
    legal = client.get(f'/api/cs/position/{sid}').json()['legal']
    bad = next(c for c in range(52) if c not in legal)
    good = ({'cards': legal[:3]} if aw['phase'] == 'pass'
            else {'card': legal[0]})
    badbody = ({'cards': [bad] + legal[:2]} if aw['phase'] == 'pass'
               else {'card': bad})
    r = client.post(f'/api/cs/play/{sid}',
                    json={'seat': aw['seat'], **badbody})
    assert r.status_code == 400, f'illegal card accepted ({r.status_code})'
    if aw['phase'] == 'pass':
        r = client.post(f'/api/cs/play/{sid}',
                        json={'seat': aw['seat'], 'cards': legal[:2]})
        assert r.status_code == 400, 'a 2-card pass was accepted'
    r = client.post(f'/api/cs/play/{sid}',
                    json={'seat': st['your_seat'], **good})
    assert r.status_code == 409, f'human seat accepted as an AI move ({r.status_code})'
    r = client.post(f'/api/cs/play/{sid}',
                    json={'seat': aw['seat'], 'k': aw['k'], 'ms': 800,
                          'engine': {'ver': 17, 'ep': 'webgpu'}, **good})
    assert r.status_code == 200, f'a legal AI move was rejected: {r.text}'


@test('gameplay')
def client_search_never_writes_to_the_trusted_log():
    """Quarantine, checked at the only place it can be broken."""
    log = os.environ['HEARTS_LOG_PATH']
    before = os.path.getsize(log) if os.path.exists(log) else 0
    st = _cs_new()
    sid = st['sid']
    # A whole deal is 64 actions and costs no inference at all in this mode -
    # the "AI" move is supplied, so no forward pass ever runs.
    for _ in range(80):
        cur = client.get(f'/api/state/{sid}').json()
        if cur.get('finished') or cur.get('deal_no', 1) > 1:
            break
        aw = cur.get('awaiting')
        if aw:
            legal = client.get(f'/api/cs/position/{sid}').json()['legal']
            body = ({'cards': legal[:3]} if aw['phase'] == 'pass'
                    else {'card': legal[0]})
            client.post(f'/api/cs/play/{sid}',
                        json={'seat': aw['seat'], **body})
        elif cur.get('legal'):
            client.post(f'/api/play/{sid}', json={'card': cur['legal'][0]})
        else:
            break
    after = os.path.getsize(log) if os.path.exists(log) else 0
    assert after == before, (
        'a client-search session appended to match_logs - it can now reach '
        'the leaderboard, progress stats and the training corpus')

    # ...and it DID land in the separate store, where review can find it.
    rows = [json.loads(l) for l in
            open(os.environ['HEARTS_CS_LOG_PATH'], encoding='utf-8')
            if sid in l]
    assert rows, 'a completed client-search deal wrote no cs log line'
    assert all(r['trust'] == 'client-search' for r in rows), (
        'a cs log line is missing its trust marker - a stray copy of this '
        'file would no longer identify itself')
    assert server._log_lines_for(sid), (
        '_log_lines_for cannot resolve a client-search match, so review and '
        'history are broken for this mode')


@test('gameplay')
def only_the_host_can_see_a_search_tables_ai_hands():
    """At a table the disclosure is not the player's own problem: the
    host's browser holds the AI seats' cards, which is information the
    other humans do not have, in a live game against them. So the route is
    host-only, and every seat is told search is on."""
    host = client.post('/api/identity/new', json={}).json()['key']
    guest = client.post('/api/identity/new', json={}).json()['key']
    code = client.post('/api/table/new',
                       json={'pid': host, 'client_search': True}).json()['code']
    client.post('/api/table/join', json={'code': code, 'pid': guest})
    client.post('/api/table/start', json={'code': code, 'pid': host,
                                          'timer_s': 0, 'speed': 'fast'})
    gv = client.get(f'/api/table/state/{code}?pid={guest}').json()
    assert gv['client_search'] is True, (
        'a guest is not told the host is searching, so they cannot know '
        'the host can see the AI hands')
    r = client.get(f'/api/cs/table/position/{code}?pid={guest}')
    assert r.status_code == 403, (
        f'a guest reached the AI hands ({r.status_code}) - one player '
        'seeing cards another cannot, in a game against them')
    plain = client.post('/api/table/new', json={'pid': host}).json()['code']
    r = client.get(f'/api/cs/table/position/{plain}?pid={host}')
    assert r.status_code == 409, (
        f'a NON-search table served the AI hands ({r.status_code})')


@test('gameplay')
def a_finished_client_search_match_can_be_reviewed():
    """Review gates on _finished_matches, and only the trusted-log indexer
    fills it - so until cs_log_line registered them too, a completed
    client-search match answered 'the review opens when the match ends'
    forever. Caught driving a real match, 2026-08-17."""
    fake = 'CSREVIEWSID'
    server.cs_log_line({'v': 1, 'kind': 'cs_match', 'trust': 'client-search',
                        'sid': fake, 'pid': None})
    assert (fake, 1) in server._finished_matches, (
        'a completed client-search match is not registered as finished, so '
        'its review is permanently unreachable')


@test('gameplay')
def practice_eval_returns_a_ranking():
    sid = client.post('/api/new', json={'practice': True}).json()['sid']
    r = client.get(f'/api/practice/eval?sid={sid}')
    assert r.status_code == 200, f'practice eval failed: {r.status_code}'
    d = r.json()
    assert 'top' in d and 'eq' in d, 'practice eval lost top/eq'
    assert d['n_legal'] > 0, 'practice eval reported no legal moves'


@test('gameplay')
def practice_is_never_logged():
    """PRACTICE never logs - that is the whole no-recording promise."""
    log = os.environ['HEARTS_LOG_PATH']
    before = os.path.getsize(log) if os.path.exists(log) else 0
    sid = client.post('/api/new', json={'practice': True}).json()['sid']
    st = client.get(f'/api/state/{sid}').json()
    for _ in range(3):
        cur = client.get(f'/api/state/{sid}').json()
        if not cur.get('legal'):
            break
        client.post(f'/api/play/{sid}', json={'card': cur['legal'][0]})
    after = os.path.getsize(log) if os.path.exists(log) else 0
    assert after == before, 'a practice session wrote to the match log'


# ===========================================================================
# access control - what the public must never reach
# ===========================================================================

@test('access')
def admin_is_localhost_only():
    """The guard is structural (loopback AND no CF header), which is why
    the whole feature can live in a public repo."""
    assert client.get('/admin').status_code == 200, (
        'admin page unreachable from localhost - the ops window is broken')
    assert client.get('/admin', headers=TUNNEL).status_code == 403, (
        'ADMIN PAGE REACHABLE THROUGH THE TUNNEL - public exposure')
    assert client.get('/api/admin/status', headers=TUNNEL).status_code == 403, (
        'ADMIN STATUS API REACHABLE THROUGH THE TUNNEL - public exposure')


@test('access')
def admin_status_reports_the_expected_shape():
    d = client.get('/api/admin/status').json()
    for key in ('now', 'logs', 'system', 'backup'):
        assert key in d, f'admin status lost its "{key}" section'
    assert 'solo_live' in d['now']


@test('access')
def a_bad_share_token_is_refused():
    r = client.get('/api/review?share=not-a-real-token.deadbeef')
    assert r.status_code in (403, 404), (
        f'a forged share token returned {r.status_code}')


# ===========================================================================
# consistency - files that must agree with each other
# ===========================================================================

def _client_emotes():
    """Parse the EMOTES mirror out of index.html."""
    src = read(os.path.join(STATIC, 'index.html'))
    block = re.search(r'const EMOTES = \{(.*?)\n\};', src, re.S)
    assert block, 'could not find the EMOTES object in index.html'
    return dict(re.findall(r"(\w+):\s*'([^']+)'", block.group(1)))


@test('consistency')
def emote_whitelist_matches_the_client_mirror():
    """server.py's EMOTES is the enforcement list; index.html mirrors it.

    They are edited together by hand, so they drift - and a drift means
    either an emote nobody can send or a button that 400s.
    """
    server_side = dict(server.EMOTES)
    client_side = _client_emotes()
    assert server_side == client_side, (
        'EMOTES drifted between server.py and index.html\n'
        f'  server-only: {sorted(set(server_side) - set(client_side))}\n'
        f'  client-only: {sorted(set(client_side) - set(server_side))}\n'
        '  differing: ' + str(sorted(
            k for k in set(server_side) & set(client_side)
            if server_side[k] != client_side[k])))


@test('consistency')
def supporter_emote_sets_match():
    src = read(os.path.join(STATIC, 'index.html'))
    block = re.search(r'const SUP_EMOTES = new Set\(\[(.*?)\]\);', src, re.S)
    assert block, 'could not find SUP_EMOTES in index.html'
    client_sup = set(re.findall(r"'([^']+)'", block.group(1)))
    assert set(server.SUP_EMOTES) == client_sup, (
        'the supporter-only emote set drifted between server and client\n'
        f'  server-only: {sorted(set(server.SUP_EMOTES) - client_sup)}\n'
        f'  client-only: {sorted(client_sup - set(server.SUP_EMOTES))}')


@test('consistency')
def every_emote_has_an_svg():
    missing = []
    for name, val in server.EMOTES.items():
        if re.fullmatch(r'[0-9a-f-]+', val):       # codepoint, not a phrase
            p = os.path.join(STATIC, 'emotes', f'{val}.svg')
            if not os.path.exists(p):
                missing.append(f'{name} -> {val}.svg')
    assert not missing, 'emotes with no artwork on disk: ' + ', '.join(missing)


@test('consistency')
def client_search_discloses_what_it_is_doing():
    """The mode's whole defensibility is that it says what it is. It runs
    the AI on the visitor's hardware, which means their machine holds the
    AI's cards - so the page must state that it is unranked AND why, and
    must not claim the passes are searched when they are not."""
    js = read(os.path.join(STATIC, 'cs_play.js'))
    idx = read(os.path.join(STATIC, 'index.html'))
    assert 'cs_play.js' in idx, 'index.html no longer loads the driver'
    assert 'id="cs-status"' in idx, 'the disclosure line has no mount point'
    assert 'window.csHost' in idx, (
        'the csHost bridge is gone; cs_play.js cannot apply state, and '
        'assigning window.state would silently write a different variable')
    low = js.lower()
    assert 'unranked' not in low and 'unranked' not in idx.lower(), (
        '"unranked" is back - it implies a ranking system this site does '
        'not have (user call 2026-08-17); say what is true instead, that '
        'these matches stay off the leaderboard')
    assert 'off the leaderboard' in low, (
        'the driver no longer tells the player these matches do not count')
    assert "ai’s cards" in low or "ai's cards" in low, (
        'the driver states the restriction without the reason - players '
        'should learn WHY it does not count')
    assert 'raw policy' not in low, (
        'the "passes use the raw policy" caveat is back - since engine VER '
        '17 (an_choose_pass) the client searches passes at K=24 too, so '
        'that text now understates what the device is doing')
    assert 'passes' in low, (
        'the note no longer mentions passes at all; the player should know '
        'the device is choosing those as well')
    # every seat at a searching table is warned, not just the host
    assert 'host’s device' in low or "host's device" in low, (
        'guests at a searching table are no longer told the host can see '
        'the AI hands - that is information they do not have, in a live '
        'game against them')
    # a desynced search must never be played
    assert 'desync' in low and 'reject' in low, (
        'cs_play.js no longer refuses a desynced search result')

    # the toggle contract: one switch across the modes, daily excluded
    assert 'id="cs-toggle"' in idx, 'the search toggle is gone'
    low_idx = idx.lower()
    assert 'off the leaderboard' in low_idx, (
        'the toggle no longer says these games do not count')
    assert 'your computer' in low_idx or 'your browser' in low_idx, (
        'the toggle no longer says whose machine does the work')
    # the note must not be able to resize the menu around it
    assert re.search(r'#home \.opanel[^}]*max-width', idx, re.S), (
        'the menu panel lost its width cap, so opening the search note '
        'stretches the whole menu sideways again')
    assert re.search(r'd\.disabled\s*=\s*on', idx), (
        'the daily challenge is no longer disabled while search is on - it '
        'is a ranked board and a search match cannot count for it')

    # whole-match replay: a single-deal load passes in the wrong direction
    # from deal 2 on (pass_direction = deal_count % 4)
    assert 'deals' in js and 'deal_idx' in js, (
        'cs_play.js is no longer sending every deal to the worker; pass '
        'direction will be wrong from deal 2 and the search will pick '
        'cards the seat does not hold')


@test('consistency')
def every_log_replay_installs_the_logged_hands():
    """Seed dealing is toolchain-bound (std::shuffle), which is why deal
    lines carry their hands. A replay that trusts the seed instead
    desyncs and steps cards that are not in the current player's hand;
    hands grew to 14, and the engine's GetLegalActions fills a
    std::array<int,13> with an unchecked idx++, so the canary died and
    the whole server aborted with 'stack smashing detected' - 11 times
    between 2026-08-12 and 2026-08-17, reachable by any client that
    could POST /api/search/upload.

    Every function replaying logged actions must install the hands AND
    guard the desync: the install prevents it, the guard turns a missed
    case into a 500 instead of a dead process."""
    import inspect
    for fn in (server.compute_review, server._legal_deals,
               server._decision_replay, server.compute_insight,
               server.compute_match_stats):
        src = inspect.getsource(fn)
        assert '_apply_logged_hands' in src, (
            f'{fn.__name__} replays logged actions without installing the '
            'logged hands - it will desync on a foreign toolchain and can '
            'overflow the engine legal-action buffer')
        assert 'get_current_player() !=' in src, (
            f'{fn.__name__} has no replay-desync guard - a desync there '
            'aborts the process instead of failing one request')


@test('consistency')
def a_restored_table_has_every_field_it_will_be_asked_for():
    """_restore_tables builds tables with Table.__new__, so __init__
    never runs; it then sets only _SNAP_FIELDS plus a small defaults
    block. Any attribute __init__ defines that is in NEITHER list is
    simply ABSENT on a table restored after a restart, and the first
    request touching it 500s - silently, because only tables that
    survived a restart are affected.

    This is a whole-class guard, not a one-field one: hearts_broken
    shipped broken this way on 2026-08-17, and the three spectator
    fields had been missing for longer. Adding a field to Table means
    deciding whether it SURVIVES a restart (_SNAP_FIELDS) or RESETS
    (the defaults block) - this fails until you have."""
    import inspect
    # [a-z0-9_]: t0 is a field name, and a pattern that quietly skipped
    # it would be a blind spot in the guard rather than in the code
    defaults = set(re.findall(r'^\s+t\.([a-z0-9_]+)\s*=',
                              inspect.getsource(server._restore_tables), re.M))
    fields = set(re.findall(r'^\s+self\.([a-z0-9_]+)\s*=',
                            inspect.getsource(server.Table.__init__), re.M))
    gap = sorted(fields - set(server._SNAP_FIELDS) - defaults)
    assert not gap, (
        f'never set on a restored table, will AttributeError when read: '
        f'{gap} - add each to _SNAP_FIELDS (survives the restart) or to '
        'the defaults block in _restore_tables (resets cleanly)')


@test('consistency')
def hearts_broken_is_tracked_and_shipped_everywhere():
    """One flag, two writers (Session._apply, Table._apply) and three
    payloads (solo state, table state, spectator snapshot). Any client
    that does not receive it silently reverts to deriving the badge from
    plays it watched, which is the bug this replaced."""
    src = read(os.path.join(WEB, 'server.py'))
    n = src.count("'hearts_broken':")
    assert n >= 3, (
        f'only {n} state payload(s) carry hearts_broken - the solo state, '
        'the table state and the spectator snapshot all need it')
    w = src.count('self.hearts_broken = True')
    assert w == 2, (
        f'{w} writer(s) set hearts_broken - Session._apply and '
        'Table._apply must each mark the suit broken')
    js = read(os.path.join(STATIC, 'index.html'))
    body = re.search(r'function render\(\) \{(.*?)\n\}', js, re.S)
    assert body, 'render() vanished from index.html'
    assert 'hearts_broken' in body.group(1), (
        'render() no longer adopts state.hearts_broken - refresh, resume '
        'and mid-deal join go back to guessing the badge')


@test('consistency')
def nav_cache_version_is_uniform():
    """One stale ?v= means that page keeps serving an old menu."""
    versions = {}
    for name in os.listdir(STATIC):
        if not name.endswith('.html'):
            continue
        for v in re.findall(r'nav\.js\?v=(\d+)', read(os.path.join(STATIC, name))):
            versions.setdefault(v, []).append(name)
    assert len(versions) <= 1, (
        'nav.js cache versions disagree, so some pages serve a stale menu: '
        + json.dumps(versions))


@test('consistency')
def referenced_static_assets_exist():
    """A renamed asset that some page still points at 404s in production."""
    missing = []
    for name in os.listdir(STATIC):
        if not name.endswith('.html'):
            continue
        src = read(os.path.join(STATIC, name))
        for ref in set(re.findall(r'/static/([A-Za-z0-9_./-]+\.'
                                  r'(?:json|woff2|wasm|svg|png|css|js))(?![A-Za-z0-9])', src)):
            if not os.path.exists(os.path.join(STATIC, ref)):
                missing.append(f'{name} -> /static/{ref}')
    assert not missing, 'referenced but absent: ' + ', '.join(sorted(missing))


# ===========================================================================
# layout guards - invariants behind bugs that already bit us once
# ===========================================================================

@test('layout')
def review_mobile_body_is_not_a_fixed_height_flex_column():
    """2026-08-13: the desktop body is height:100% + display:flex, and the
    mobile pass overrode only `overflow`. Every child kept flex-shrink:1,
    so opening a panel COMPRESSED the header and note above it."""
    src = read(os.path.join(STATIC, 'review.html'))
    block = re.search(r'body\.mobile \{(.*?)\}', src, re.S)
    assert block, 'the body.mobile rule vanished from review.html'
    assert 'height:auto' in block.group(1).replace(' ', ''), (
        'review.html body.mobile no longer releases the fixed height - '
        'expanding a panel will compress the header again')


@test('layout')
def nav_dropdown_escapes_the_band_stacking_context():
    """2026-08-13: the dropdown is a child of #siteband (z-index 55) while
    review.html's #top is 56, so the menu opened UNDER that page's
    buttons. A child cannot outrank its parent's stacking context."""
    src = read(os.path.join(STATIC, 'nav.js'))
    # anchored to line start: a COMMENTED-OUT call still contains the
    # substring, which let a deliberately broken build pass once
    assert re.search(r'(?m)^\s*document\.body\.appendChild\(dd\)', src), (
        'the nav dropdown is no longer reparented to <body>; it will open '
        'underneath any page header with a higher z-index')
    assert re.search(r"dd\.style\.zIndex\s*=\s*'(\d+)'", src), (
        'the reparented dropdown lost its explicit z-index')
    z = int(re.search(r"dd\.style\.zIndex\s*=\s*'(\d+)'", src).group(1))
    top_z = int(re.search(r'#top \{[^}]*z-index:(\d+)',
                          read(os.path.join(STATIC, 'review.html')),
                          re.S).group(1))
    assert z > top_z, f'dropdown z-index {z} no longer beats review #top {top_z}'


@test('layout')
def game_menu_paints_over_the_header_controls():
    """The open dropdown must cover the game page's control row."""
    src = read(os.path.join(STATIC, 'index.html'))
    menu_z = int(re.search(r'#menu-btn-wrap \{[^}]*z-index:(\d+)', src, re.S).group(1))
    ctl_z = int(re.search(r'#theme-toggle \{[^}]*z-index:(\d+)', src, re.S).group(1))
    assert menu_z > ctl_z, (
        f'menu z-index {menu_z} no longer beats the controls {ctl_z}')


@test('layout')
def supporter_badge_keeps_its_dark_mount():
    """The gold moon washes out on the gold active-turn pill without it."""
    src = read(os.path.join(STATIC, 'index.html'))
    block = re.search(r'(?m)^\s*\.supbadge \{(.*?)\}', src, re.S)
    assert block and 'border-radius:50%' in block.group(1).replace(' ', ''), (
        'the supporter badge lost its circular dark mount')


@test('layout')
def the_slow_review_paths_show_a_loading_indicator():
    """A cold review costs seconds of server CPU (_review_get_or_compute
    falls through memory and disk to compute_review), and the end-screen
    insight replays the match the same way. Both used to sit blank
    through it and look finished/broken. What actually matters here is
    not that a spinner exists but that it is CLEARED on every exit -
    including the error path, where a stranded spinner outlives the
    thing it was waiting for."""
    rev = read(os.path.join(STATIC, 'review.html'))
    assert 'id="rev-loading"' in rev and '#rev-loading.on' in rev, (
        'the review page lost its loading overlay (markup or CSS)')
    boot = rev[rev.index('// ---- boot'):]
    assert boot.count('doneLoading()') >= 2, (
        'the review loader is not cleared on BOTH the success and error '
        'paths - a failed review would spin forever')
    idx = read(os.path.join(STATIC, 'index.html'))
    m = re.search(r'function showMatchEnd\s*\(', idx)
    assert m, 'showMatchEnd() vanished from index.html'
    body = idx[m.end():]
    nxt = re.search(r'\n(?:async )?function ', body)
    if nxt:
        body = body[:nxt.start()]
    assert 'vring' in body, (
        'the end-screen insight card no longer holds its slot while the '
        'server replays the match')
    assert body.count('dropHold()') >= 2, (
        'the end-screen spinner is not dropped on both the resolved and '
        'the failed path')


@test('layout')
def end_screen_score_tables_are_ranked_not_seat_ordered():
    """Score tables read top-to-bottom as the standing: 1st place first,
    4th last. The deal screen has no placements yet so it ranks on the
    running totals (LOWEST wins at Hearts); both match-end tables - the
    ranked one and the practice one - rank on the engine's placements."""
    src = read(os.path.join(STATIC, 'index.html'))
    assert 'const byPlace =' in src, (
        'the shared ranking helper vanished; the tables below are only '
        'ordered if something still sorts them')
    for fn, key in (('showDealEnd', 'byPlace(ev.totals)'),
                    ('showMatchEnd', 'byPlace(state.placements)')):
        m = re.search(r'function %s\s*\(' % fn, src)
        assert m, f'{fn}() vanished from index.html'
        body = src[m.end():]
        nxt = re.search(r'\n(?:async )?function ', body)
        if nxt:
            body = body[:nxt.start()]
        assert key in body, (
            f'{fn}() no longer ranks its score table with {key}')
        assert not re.search(r'\[0, ?1, ?2, ?3\]\.map', body), (
            f'{fn}() is emitting a score row per seat in SEAT order again')


@test('layout')
def hearts_broken_badge_resets_on_every_game_entry():
    """2026-08-15: heartsBroken is client-only state with no server
    mirror. It was reset at deal_end and on a table match_no change -
    neither of which fires when a game STARTS - so a badge lit in the
    previous game rode onto the opening felt of the next (reported:
    solo -> break hearts -> host table -> close table -> solo). The
    lobby is why entering a table needs its own reset: handleTableState
    returns early for state 'lobby', above the match_no reset."""
    src = read(os.path.join(STATIC, 'index.html'))
    for fn in ('newMatch', 'enterTable', 'startSpectate'):
        m = re.search(r'(?:async )?function %s\s*\(' % fn, src)
        assert m, f'{fn}() vanished from index.html'
        # slice to the next top-level function so a NEIGHBOUR's reset can
        # never satisfy this one
        body = src[m.end():]
        nxt = re.search(r'\n(?:async )?function ', body)
        if nxt:
            body = body[:nxt.start()]
        assert 'setHeartsBroken(false)' in body, (
            f'{fn}() no longer resets the hearts-broken badge - a badge '
            'lit in the previous game will ride into the next one')


# ===========================================================================
# feedback - the public write path, so it gets the closest scrutiny
# ===========================================================================

@test('feedback')
def feedback_page_serves():
    r = client.get('/feedback')
    assert r.status_code == 200 and b'Send feedback' in r.content


@test('feedback')
def a_report_is_stored_with_server_side_context():
    before = _feedback_rows()
    r = client.post('/api/feedback', json={
        'message': 'the review panel showed the 4C grouped with the 6C',
        'category': 'bug', 'page': '/review?share=abc', 'sid': 'TESTSID'})
    assert r.status_code == 200, f'submit failed: {r.status_code} {r.text}'
    rows = _feedback_rows()
    assert len(rows) == len(before) + 1, 'the report was not appended'
    e = rows[-1]
    assert e['category'] == 'bug' and e['sid'] == 'TESTSID'
    assert e['model'], 'the serving model hash was not attached'
    assert 'ua' in e, 'the user agent was not attached'
    assert 'ip' not in e and 'cf-connecting-ip' not in e, (
        'an IP address was stored - the site promises it is not')


@test('feedback')
def the_honeypot_swallows_bots():
    before = _feedback_rows()
    r = client.post('/api/feedback', json={
        'message': 'buy cheap watches', 'website': 'http://spam.example'})
    assert r.status_code == 200, 'the honeypot response should look normal'
    assert len(_feedback_rows()) == len(before), (
        'a honeypot-tripping submission was stored anyway')


@test('feedback')
def junk_submissions_are_refused():
    assert client.post('/api/feedback', json={'message': 'hi'}).status_code == 400
    assert client.post('/api/feedback',
                       json={'message': 'x' * 5000}).status_code == 400


@test('feedback')
def feedback_never_touches_the_match_log():
    log = os.environ['HEARTS_LOG_PATH']
    before = os.path.getsize(log) if os.path.exists(log) else 0
    client.post('/api/feedback', json={'message': 'a perfectly fine report'})
    after = os.path.getsize(log) if os.path.exists(log) else 0
    assert after == before, 'a feedback submission wrote to the match log'


@test('feedback')
def the_fallback_address_is_only_offered_when_configured():
    d = client.get('/api/feedback/meta').json()
    assert 'email' in d, '/api/feedback/meta lost its email field'
    if d['email'] is None:
        src = read(os.path.join(STATIC, 'feedback.html'))
        assert '@perilune.ai' not in src, (
            'the page hardcodes an address while none is configured - it '
            'would advertise a dead mailbox')


@test('privacy')
def visitor_counter_dedupes_and_stores_no_address():
    """Counts people, not requests - and keeps nothing identifying.

    The map is in-memory only, keyed by an HMAC under an ephemeral
    per-process salt. A raw IP appearing anywhere in it (or the count
    rising per REQUEST rather than per visitor) is a privacy failure.
    """
    server._visits.clear()
    ip_a, ip_b = '198.51.100.4', '203.0.113.9'
    for _ in range(5):                       # one visitor, many requests
        client.get('/about', headers={'CF-Connecting-IP': ip_a})
    assert server._active_visitors() == 1, (
        'the counter counts requests, not visitors')
    client.get('/how', headers={'CF-Connecting-IP': ip_b})
    assert server._active_visitors() == 2, 'a second visitor was not counted'

    blob = repr(list(server._visits.keys()))
    assert ip_a not in blob and ip_b not in blob, (
        'A RAW IP IS HELD IN MEMORY - the counter must store only salted '
        'digests')
    assert all(isinstance(k, bytes) for k in server._visits), (
        'visitor keys are not opaque digests')


@test('privacy')
def visitor_counter_is_never_persisted():
    """Nothing about visitors may reach disk or the backup set."""
    import hearts_web.backup_sync as bs
    assert not any('visit' in f for f in bs.DATA_FILES), (
        'a visitor file entered the backup set')
    for name in os.listdir(WEB):
        assert 'visit' not in name.lower(), (
            f'{name} looks like a persisted visitor file')


@test('privacy')
def the_uptime_robot_is_not_counted_as_a_person():
    server._visits.clear()
    client.get('/api/leaderboard', headers={
        'CF-Connecting-IP': '192.0.2.50', 'User-Agent': 'curl/8.4.0'})
    assert server._active_visitors() == 0, (
        'the uptime monitor is being counted as a visitor')


# ===========================================================================
# runner
# ===========================================================================

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ''
    selected = [t for t in TESTS if only in t.group or only in t.__name__]
    if not selected:
        print(f'no tests match "{only}"')
        return 1
    passed, failed = 0, []
    group = None
    for t in selected:
        if t.group != group:
            group = t.group
            print(f'\n[{group}]')
        try:
            t()
            print(f'  PASS  {t.__name__}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {t.__name__}')
            for line in str(e).splitlines() or ['(no message)']:
                print(f'        {line}')
            if not isinstance(e, AssertionError):
                print('        ' + traceback.format_exc().splitlines()[-1])
            failed.append(t.__name__)
    print(f'\n{passed} passed, {len(failed)} failed')
    if failed:
        print('failed: ' + ', '.join(failed))
    for tmp in (_tmp_log.name, _tmp_fb.name):
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())


@test('consistency')
def client_search_seeds_the_review_cache_on_the_agreed_contract():
    """Every client-search decision is a real K=64 search at a position the
    review will re-analyse anyway, so it is written into the store
    review.html reads. Three files share this contract - review.html owns
    it, verify_end.js and cs_play.js write into it - and a mismatch in any
    of schema / db / store / key silently produces a cache that is never
    read, which looks exactly like it working."""
    rev = read(os.path.join(STATIC, 'review.html'))
    ver = read(os.path.join(STATIC, 'verify_end.js'))
    cs = read(os.path.join(STATIC, 'cs_play.js'))
    schema = re.search(r"DEEP_SCHEMA\s*=\s*'([^']+)'", rev).group(1)
    for name, src in (('verify_end.js', ver), ('cs_play.js', cs)):
        assert re.search(r"SCHEMA\s*=\s*'%s'" % re.escape(schema), src), (
            f'{name} does not write review.html DEEP_SCHEMA {schema!r}; its '
            'entries will never be read')
        assert "'perilune-review'" in src and "'evals'" in src, (
            f'{name} lost the review db/store name')
        assert re.search(r'\$\{\w+\}\|\$\{\w+\(?\)?\}\|\$\{\w+\}\|'
                         r'\$\{[\w.]+\}\|\$\{[\w.]+\}\|\$\{[\w.]+\}', src), (
            f'{name} no longer builds the 6-part schema|ident|md5|K|d|i key')
    # the review indexes plays only; the engine indexes the full action
    # list, so the writer has to subtract the pass offset
    assert 'pass_offset' in cs, (
        'cs_play.js is not subtracting the pass offset, so every entry on a '
        'passing deal is filed 12 positions off and never matches')


@test('consistency')
def every_analysis_worker_url_is_versioned_and_agrees():
    """The worker is loaded from four places and cached by URL. An
    UNVERSIONED load serves whatever the browser already has: on
    2026-08-17 cs_play.js had no ?v=, so a stale worker with no 'livepass'
    kind fell through its dispatch chain to _an_analyze with an undefined
    actionIdx, analysed a play at a passing position, and the engine's -1
    surfaced as "search desync".

    The worker also picks the ENGINE url from its own VER, so a stale
    worker pins a stale wasm too. Every caller must carry ?v=, and they
    must all carry the SAME one - a split means two engine versions live
    in one browser."""
    seen = {}
    for name in ('cs_play.js', 'verify_end.js', 'index.html', 'review.html'):
        src = read(os.path.join(STATIC, name))
        for m in re.finditer(r"new Worker\('([^']*analysis_worker\.js[^']*)'",
                             src):
            url = m.group(1)
            ver = re.search(r'\?v=(\d+)', url)
            assert ver, f'{name} loads the worker unversioned: {url}'
            seen.setdefault(ver.group(1), []).append(name)
    assert seen, 'no analysis_worker.js loads found at all'
    assert len(seen) == 1, (
        f'analysis_worker.js is loaded at more than one version: '
        f'{ {k: v for k, v in seen.items()} } - one page will run a '
        'different worker, and therefore a different engine')


@test('consistency')
def cs_play_cache_version_matches_the_file():
    """cs_play.js is cached by its ?v=. It sat at v=1 through a dozen
    rewrites on 2026-08-17 and browsers kept serving the FIRST version -
    which had no pass handling, so it asked the engine for a play at a
    passing position and the refusal surfaced as "search desync". The
    source was right the whole time; the bytes being executed were not.

    Binding the pin to a constant inside the file means the two cannot
    drift without this failing."""
    js = read(os.path.join(STATIC, 'cs_play.js'))
    idx = read(os.path.join(STATIC, 'index.html'))
    inner = re.search(r'const CS_VER\s*=\s*(\d+)', js)
    assert inner, 'cs_play.js lost its CS_VER marker'
    pin = re.search(r'cs_play\.js\?v=(\d+)', idx)
    assert pin, 'index.html loads cs_play.js without a ?v= cache key'
    assert inner.group(1) == pin.group(1), (
        f'cs_play.js declares CS_VER={inner.group(1)} but index.html pins '
        f'?v={pin.group(1)} - browsers will run whichever they cached')


@test('consistency')
def searched_ai_moves_still_obey_the_speed_setting():
    """A forced move costs no search at all - the engine steps a single
    legal card with no net call - and the whole last trick is forced. So
    without a dwell the endgame snapped after twelve tricks of visible
    thinking (reported 2026-08-17). The driver waits out the remainder of
    pace().ai, which is 0 at 'instant' - so instant still plays the moment
    the search lands."""
    js = read(os.path.join(STATIC, 'cs_play.js'))
    idx = read(os.path.join(STATIC, 'index.html'))
    assert not re.search(r'H\.paceAi\(\)\s*-\s*\(Date\.now', js), (
        'the driver is padding searched moves against pace().ai again. '
        'animate() already sleeps one pace().ai after every AI card, so '
        'this makes a searched move take TWO beats where a non-search '
        'game takes one (user call 2026-08-17: single beat)')
