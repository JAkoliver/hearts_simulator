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
    try:
        os.unlink(_tmp_log.name)
    except OSError:
        pass
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
