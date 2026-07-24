"""Head-to-head model comparison instrument for Hearts.

Compares exactly TWO models (newest vs oldest .pth in eval_models/, by mtime)
using paired duplicate deals in both directions:

  Direction 1: 1 newer vs 3 older, against an all-older table on the SAME deal
  Direction 2: 1 older vs 3 newer, against an all-newer table on the SAME deal

Every deal is played on four synced tables (the engine's RNG is consumed only
by reset()'s shuffle, so envs built from the same seed deal identically), which
removes deal/seat luck from the score comparison and gives each model exactly
8 seat-games per deal in a balanced mix of weak- and strong-table contexts.

Outputs: paired score differentials with CIs and p-values, per-game outcome
decomposition, tactical telemetry with Wilson CIs and significance-starred
deltas, a policy divergence probe on identical states, and a CSV history row
per model for tracking progress across generations.

The previous multi-model (1/3/4 seat) analyzer is preserved in analyzer_legacy.py.
"""

import os
import glob
import heapq
import random
import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import tqdm

import hearts_env
from hearts_net import HeartsNet

# ----------------------------- configuration -----------------------------
PAIRED_DEALS = 2000     # deals per run; 4 games are played per deal
MODE = 'argmax'         # 'argmax' (deterministic) or 'sample' (stochastic)
SEED = 42               # deal sequence + torch/random seed (sample mode)
TOP_DISAGREEMENTS = 5   # highest-divergence states to print
HISTORY_CSV = 'analyzer_history.csv'
PASSING = True          # play with the card-passing phase (set False to
                        # compare pre-passing models on their own terms)

QS, KS, AS = 36, 37, 38
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUITS = ['C', 'D', 'S', 'H']

# ----------------------------- model loading -----------------------------

class LegacyHeartsNet(nn.Module):
    """v1 architecture (181 -> 256 -> 256 MLP) so old-lineage checkpoints
    remain loadable for cross-generation comparisons."""

    def __init__(self):
        super(LegacyHeartsNet, self).__init__()
        self.shared_fc1 = nn.Linear(181, 256)
        self.shared_fc2 = nn.Linear(256, 256)
        self.policy_head = nn.Linear(256, 52)
        self.value_head = nn.Linear(256, 1)

    def forward(self, observation, legal_actions_mask):
        # v1 models predate the passing extension: feed them the first 181
        # dims (layout-identical to their training obs). They can still act
        # in passing states, just blindly - flagged in the report.
        x = F.relu(self.shared_fc1(observation[..., :181]))
        x = F.relu(self.shared_fc2(x))
        state_value = self.value_head(x)
        logits = self.policy_head(x)
        masked_logits = logits.masked_fill(~legal_actions_mask, float('-inf'))
        return masked_logits, state_value

class TruncateObs(nn.Module):
    """Feed an older-generation net the observation prefix it was trained on.
    The observation layout is prefix-stable across engine versions."""

    def __init__(self, net, obs_dim):
        super(TruncateObs, self).__init__()
        self.net = net
        self.obs_dim = obs_dim

    def forward(self, observation, legal_actions_mask):
        return self.net(observation[..., :self.obs_dim], legal_actions_mask)

def load_model(path):
    """Load a checkpoint from any lineage, auto-detecting its architecture."""
    sd = torch.load(path, weights_only=True)
    if 'shared_fc1.weight' in sd:
        net = LegacyHeartsNet()  # v1 MLP; slices to 181 dims internally
        net.load_state_dict(sd)
        net.eval()
        return net, "v1"

    if 'card_embed.weight' in sd:
        # v5 card-token transformer (2026-07-15+); net_from_checkpoint infers
        # width/depth. Same forward(obs, mask) surface as every other lineage.
        from hearts_net import net_from_checkpoint
        net = net_from_checkpoint(path)
        net.eval()
        return net, "v5"

    obs_dim = sd['input_fc.weight'].shape[1]
    net = HeartsNet(obs_dim=obs_dim)
    # Pre-belief checkpoints lack the belief head; leave it randomly
    # initialized (unused by forward()). Anything else missing is a real error.
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys in {path}: {unexpected}"
    # Pre-belief checkpoints lack belief_head; pre-oracle (pre-2026-07)
    # checkpoints lack oracle_fc1/2. Neither is used by forward().
    assert all(k.startswith(('belief_head.', 'oracle_fc')) for k in missing), \
        f"missing keys: {missing}"
    net.eval()
    arch = {550: "v4", 238: "v3"}.get(obs_dim, f"{obs_dim}d")
    if obs_dim < 550:
        return TruncateObs(net, obs_dim), arch
    return net, arch

# ----------------------------- small helpers -----------------------------

def card_name(c):
    return RANKS[c % 13] + SUITS[c // 13]

def card_rank(c):
    return (c % 13) + 2

def card_suit(c):
    return c // 13

def trick_points(cards):
    return sum(1 for c in cards if card_suit(c) == 3) + (13 if QS in cards else 0)

def winner_offset(trick_cards):
    """Index within the trick (play order) of the winning card."""
    led = card_suit(trick_cards[0])
    best, best_off = -1, 0
    for off, c in enumerate(trick_cards):
        if card_suit(c) == led and card_rank(c) > best:
            best, best_off = card_rank(c), off
    return best_off

def danger_score(c, queen_live):
    """Rough danger ranking of holding a card, for slough-quality scoring."""
    if c == QS:
        return 1000
    if c in (KS, AS) and queen_live:
        return 900 + card_rank(c)
    if card_suit(c) == 3:  # hearts
        return 500 + card_rank(c)
    return card_rank(c)

def wilson_ci(successes, n, z=1.96):
    """95% Wilson score interval for a proportion, in percent."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, center - half) * 100, min(1.0, center + half) * 100

def two_prop_sig(s1, n1, s2, n2, z_crit=1.96):
    """Two-proportion z-test; returns True when the rate gap is significant."""
    if n1 == 0 or n2 == 0:
        return False
    p1, p2 = s1 / n1, s2 / n2
    p = (s1 + s2) / (n1 + n2)
    var = p * (1 - p) * (1 / n1 + 1 / n2)
    if var <= 0:
        return False
    return abs(p1 - p2) / (var ** 0.5) > z_crit

# ----------------------------- action selection -----------------------------

def choose_action(network, obs, legal, mode):
    state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.zeros((1, 52), dtype=torch.bool)
    for a in legal:
        mask_tensor[0, a] = True
    with torch.no_grad():
        logits, _ = network(state_tensor, mask_tensor)
        if mode == 'sample':
            action = Categorical(logits=logits).sample().item()
        else:
            action = torch.argmax(logits, dim=1).item()
    return action, logits.squeeze(0)

def masked_entropy(logits):
    return Categorical(logits=logits.unsqueeze(0)).entropy().item()

def sym_kl(logits_p, logits_q, legal):
    p = torch.softmax(logits_p, dim=-1)[legal].clamp_min(1e-9)
    q = torch.softmax(logits_q, dim=-1)[legal].clamp_min(1e-9)
    return float((p * (p / q).log()).sum() + (q * (q / p).log()).sum())

# ----------------------------- telemetry -----------------------------

TACTIC_KEYS = [
    "qs_pass", "qs_dump", "duck", "duck4", "slough", "bleed",
    "lead_safe", "moon_def", "ak_punished", "control", "forced_win",
]

def new_telemetry():
    t = {
        "games_played": 0, "total_penalty": 0,
        "wins": 0, "clean": 0, "disaster": 0, "qs_eaten": 0,
        "moons_shot": 0, "moons_conceded": 0,
        "decisions": 0, "entropy_sum": 0.0,
        "phase_pts": [0, 0, 0],  # penalty points taken in tricks 1-4 / 5-9 / 10-13
    }
    for k in TACTIC_KEYS:
        t[k + "_opps"] = 0
        t[k + "_succ"] = 0
    return t

# ----------------------------- one game -----------------------------

def play_game(env, networks, names, telem, mode, div=None):
    """Play one round, updating telemetry per model name. Returns raw scores.

    When div is provided (dict with newer/older nets + accumulators), both
    models are also queried on every state for the policy divergence probe.
    """
    env.reset()
    done = False

    trick_cards = []
    trick_leader = None
    tricks_done = 0
    round_pts = [0, 0, 0, 0]          # raw (pre-moon) points taken this round
    voids = [[False] * 4 for _ in range(4)]
    qs_out = False                     # QS seen in a completed trick
    qs_eater = None
    pass_picks = [[] for _ in range(4)]
    had_qs_at_pass = [False] * 4

    while not done:
        p = env.get_current_player()
        t = telem[names[p]]
        obs = env.observe()
        legal = [a for a in env.get_legal_actions() if a != -1]
        hand = [i for i in range(52) if obs[i] == 1.0]

        # --- passing phase: pick a card to pass, no trick logic ---
        if env.is_passing():
            if not pass_picks[p] and QS in hand:
                had_qs_at_pass[p] = True
            action, logits = choose_action(networks[p], obs, legal, mode)
            t["decisions"] += 1
            t["entropy_sum"] += masked_entropy(logits)
            if div is not None:
                new_a, new_logits = choose_action(div["newer_net"], obs, legal, 'argmax')
                old_a, old_logits = choose_action(div["older_net"], obs, legal, 'argmax')
                div["states"][0] += 1
                if new_a != old_a:
                    div["disagree"][0] += 1
                    kl = sym_kl(new_logits, old_logits, legal)
                    info = (f"deal {div['deal_idx']}, PASS pick {len(pass_picks[p]) + 1}/3 | "
                            f"hand: {' '.join(card_name(c) for c in sorted(hand))} | "
                            f"newer: {card_name(new_a)}  older: {card_name(old_a)}")
                    heapq.heappush(div["top"], (kl, div["counter"], info))
                    div["counter"] += 1
                    if len(div["top"]) > TOP_DISAGREEMENTS:
                        heapq.heappop(div["top"])
            pass_picks[p].append(action)
            done = env.step(action).done
            continue

        if not trick_cards:
            trick_leader = p
            for i in range(4):
                telem[names[i]]["control_opps"] += 1

        led_suit = card_suit(trick_cards[0]) if trick_cards else -1
        winner_rank = -1
        for c in trick_cards:
            if card_suit(c) == led_suit:
                winner_rank = max(winner_rank, card_rank(c))

        action, logits = choose_action(networks[p], obs, legal, mode)
        a_suit, a_rank = card_suit(action), card_rank(action)

        t["decisions"] += 1
        t["entropy_sum"] += masked_entropy(logits)

        # --- policy divergence probe (both models asked about this state) ---
        if div is not None:
            new_a, new_logits = choose_action(div["newer_net"], obs, legal, 'argmax')
            old_a, old_logits = choose_action(div["older_net"], obs, legal, 'argmax')
            phase = 0 if tricks_done <= 3 else (1 if tricks_done <= 8 else 2)
            div["states"][phase] += 1
            if new_a != old_a:
                div["disagree"][phase] += 1
                kl = sym_kl(new_logits, old_logits, legal)
                info = (f"deal {div['deal_idx']}, trick {tricks_done + 1}, "
                        f"pos {len(trick_cards) + 1}/4 | "
                        f"hand: {' '.join(card_name(c) for c in sorted(hand))} | "
                        f"trick: {' '.join(card_name(c) for c in trick_cards) or '(leading)'} | "
                        f"newer: {card_name(new_a)}  older: {card_name(old_a)}")
                heapq.heappush(div["top"], (kl, div["counter"], info))
                div["counter"] += 1
                if len(div["top"]) > TOP_DISAGREEMENTS:
                    heapq.heappop(div["top"])

        # --- tactical metrics (opportunity evaluated before the action) ---

        # QS Dumping: QS is playable and safely dumpable
        if QS in legal and trick_cards:
            is_void_in_led = all(card_suit(c) != led_suit for c in hand)
            qs_can_dump = is_void_in_led or (
                led_suit == 2 and (KS in trick_cards or AS in trick_cards))
            if qs_can_dump:
                t["qs_dump_opps"] += 1
                if action == QS:
                    t["qs_dump_succ"] += 1

        # Optimal Ducking: of the ducks taken, was it the highest safe card?
        if trick_cards and a_suit == led_suit and a_rank < winner_rank:
            safe = [c for c in legal if card_suit(c) == led_suit and card_rank(c) < winner_rank]
            if safe:
                t["duck_opps"] += 1
                optimal = a_rank == max(card_rank(c) for c in safe)
                if optimal:
                    t["duck_succ"] += 1
                if len(trick_cards) == 3:  # 4th seat: perfect information
                    t["duck4_opps"] += 1
                    if optimal:
                        t["duck4_succ"] += 1

        # Slough Quality: void in led suit, did it dump its most dangerous card?
        if trick_cards and all(card_suit(c) != led_suit for c in hand) and len(legal) >= 2:
            queen_live = (obs[104 + QS] != 1.0) and (QS not in hand) and (QS not in trick_cards)
            dangers = {c: danger_score(c, queen_live) for c in legal}
            dmax = max(dangers.values())
            if dmax >= 500:  # only score it when a genuinely dangerous card is held
                t["slough_opps"] += 1
                if dangers[action] == dmax:
                    t["slough_succ"] += 1

        # Spade Bleeding: leading low spades to flush the queen
        if not trick_cards and QS not in hand:
            low_spades = [c for c in hand if card_suit(c) == 2 and c < QS]
            if len(low_spades) >= 2 and any(c in legal for c in low_spades):
                t["bleed_opps"] += 1
                if action in low_spades:
                    t["bleed_succ"] += 1

        # Lead Safety: avoid leading a suit an opponent is known void in
        if not trick_cards and tricks_done > 0:
            lead_suits = {card_suit(c) for c in legal}
            unsafe = {s for s in lead_suits
                      if any(voids[opp][s] for opp in range(4) if opp != p)}
            safe = lead_suits - unsafe
            if safe and unsafe:
                t["lead_safe_opps"] += 1
                if a_suit in safe:
                    t["lead_safe_succ"] += 1

        # Moon Defense: last to play, one opponent owns ALL points so far,
        # the trick has points — does the model take it to block the moon?
        if len(trick_cards) == 3:
            holders = [i for i in range(4) if round_pts[i] > 0]
            if (trick_points(trick_cards) > 0 and len(holders) == 1
                    and holders[0] != p and round_pts[holders[0]] >= 8):
                winning = [c for c in legal
                           if card_suit(c) == led_suit and card_rank(c) > winner_rank]
                if winning:
                    t["moon_def_opps"] += 1
                    if action in winning:
                        t["moon_def_succ"] += 1

        # Forced Win: must follow suit and every legal card heads the trick;
        # optimal play sheds the highest card while stuck
        if trick_cards and all(card_suit(c) == led_suit for c in legal):
            if min(card_rank(c) for c in legal) > winner_rank:
                t["forced_win_opps"] += 1
                if a_rank == max(card_rank(c) for c in legal):
                    t["forced_win_succ"] += 1

        # --- execute ---
        res = env.step(action)
        done = res.done

        if trick_cards and a_suit != led_suit:
            voids[p][led_suit] = True
        trick_cards.append(action)

        if len(trick_cards) == 4:
            off = winner_offset(trick_cards)
            winner = (trick_leader + off) % 4
            pts = trick_points(trick_cards)
            round_pts[winner] += pts
            wt = telem[names[winner]]
            wt["control_succ"] += 1
            wt["phase_pts"][0 if tricks_done <= 3 else (1 if tricks_done <= 8 else 2)] += pts
            if QS in trick_cards:
                qs_out = True
                qs_eater = winner
            if trick_cards[off] in (KS, AS) and (QS in trick_cards or not qs_out):
                wt["ak_punished_opps"] += 1
                if QS in trick_cards:
                    wt["ak_punished_succ"] += 1
            tricks_done += 1
            trick_cards = []
            trick_leader = None

    # --- end of round ---
    scores = env.get_round_scores()
    min_score = min(scores)
    is_moon = sorted(scores) == [0, 26, 26, 26]
    for i in range(4):
        t = telem[names[i]]
        # QS Passed: dealt the queen on a passing round - did it pass her?
        if had_qs_at_pass[i]:
            t["qs_pass_opps"] += 1
            if QS in pass_picks[i]:
                t["qs_pass_succ"] += 1
        t["games_played"] += 1
        t["total_penalty"] += scores[i]
        if scores[i] == min_score:
            t["wins"] += 1
        if scores[i] == 0:
            t["clean"] += 1
        if scores[i] >= 13:
            t["disaster"] += 1
        if qs_eater == i:
            t["qs_eaten"] += 1
        if is_moon:
            if scores[i] == 0:
                t["moons_shot"] += 1
            else:
                t["moons_conceded"] += 1
    return scores

# ----------------------------- reporting -----------------------------

def fmt_rate(succ, n):
    if n == 0:
        return "    --      "
    lo, hi = wilson_ci(succ, n)
    return f"{succ / n * 100:5.1f}% [{lo:4.1f},{hi:5.1f}]"

def print_rate_table(title, rows, tn, to, higher_is_better):
    """rows: list of (label, key). tn/to: newer/older telemetry dicts."""
    print(f"\n--- {title} ---")
    print(f"{'Metric':<16} | {'Newer':>20} {'n':>7} | {'Older':>20} {'n':>7} | {'Delta':>8}")
    print("-" * 90)
    for label, key in rows:
        sn, nn_ = tn[key + '_succ'], tn[key + '_opps']
        so, no = to[key + '_succ'], to[key + '_opps']
        rn = sn / nn_ * 100 if nn_ else 0.0
        ro = so / no * 100 if no else 0.0
        delta = rn - ro
        if not higher_is_better.get(key, True):
            delta = -delta  # report improvement as positive
        star = '*' if two_prop_sig(sn, nn_, so, no) else ' '
        note = '' if higher_is_better.get(key, True) else '  (lower raw is better)'
        print(f"{label:<16} | {fmt_rate(sn, nn_):>20} {nn_:>7} | "
              f"{fmt_rate(so, no):>20} {no:>7} | {delta:+7.1f}{star}{note}")

def main():
    print(f"Hearts Model Comparison - paired deals, mode={MODE}, seed={SEED}")
    torch.manual_seed(SEED)
    random.seed(SEED)

    if not os.path.exists("eval_models"):
        os.makedirs("eval_models")
        print("Created 'eval_models'. Place exactly 2 .pth files inside and rerun.")
        return
    model_files = glob.glob(os.path.join("eval_models", "*.pth"))
    if len(model_files) != 2:
        print(f"Paired comparison needs exactly 2 models in eval_models/ (found {len(model_files)}).")
        print("(The old multi-model analyzer is preserved in analyzer_legacy.py.)")
        return

    model_files.sort(key=os.path.getmtime)
    older_file, newer_file = model_files
    older_name, newer_name = os.path.basename(older_file), os.path.basename(newer_file)
    older_net, older_arch = load_model(older_file)
    newer_net, newer_arch = load_model(newer_file)
    print(f" Older: {older_name} [{older_arch}]")
    print(f" Newer: {newer_name} [{newer_arch}]  (by file mtime)")
    if older_arch != newer_arch or older_arch != "v4":
        print(" NOTE: pre-v4 models see a truncated observation (v1: 181 dims, no passing")
        print("       info; v3: 238 dims, no who-played-what planes). Cross-generation")
        print("       results measure them as-is on the current rules.")

    telem = {older_name: new_telemetry(), newer_name: new_telemetry()}
    div = {"newer_net": newer_net, "older_net": older_net,
           "states": [0, 0, 0], "disagree": [0, 0, 0],
           "top": [], "counter": 0, "deal_idx": 0}

    # Four tables fed identical deal streams
    envs = [hearts_env.HeartsEnv(seed=SEED, enable_passing=PASSING) for _ in range(4)]
    diffs1, diffs2 = [], []

    print(f"\nRunning {PAIRED_DEALS} paired deals (4 games each)...")
    for d in tqdm.tqdm(range(PAIRED_DEALS), desc="Paired deals", unit="deal", ncols=80):
        seat = d % 4
        div["deal_idx"] = d

        # Direction 1: newer solo at `seat` vs all-older reference table
        nets = [older_net] * 4; nets[seat] = newer_net
        names = [older_name] * 4; names[seat] = newer_name
        s_a1 = play_game(envs[0], nets, names, telem, MODE, div=div)
        s_b1 = play_game(envs[1], [older_net] * 4, [older_name] * 4, telem, MODE)
        diffs1.append(s_a1[seat] - s_b1[seat])

        # Direction 2: older solo at `seat` vs all-newer reference table
        nets = [newer_net] * 4; nets[seat] = older_net
        names = [newer_name] * 4; names[seat] = older_name
        s_a2 = play_game(envs[2], nets, names, telem, MODE)
        s_b2 = play_game(envs[3], [newer_net] * 4, [newer_name] * 4, telem, MODE)
        diffs2.append(s_a2[seat] - s_b2[seat])

    diffs1 = np.array(diffs1, dtype=np.float64)
    diffs2 = np.array(diffs2, dtype=np.float64)
    tn, to = telem[newer_name], telem[older_name]

    # ---------------- report ----------------
    print("\n" + "=" * 90)
    print(" " * 26 + "HEAD-TO-HEAD COMPARISON REPORT")
    print("=" * 90)

    print("\n--- Paired Score Differentials (negative = solo model better than reference) ---")
    for label, diffs, alt in [
        (f"1 newer vs 3 older", diffs1, 'less'),      # newer better => diff < 0
        (f"1 older vs 3 newer", diffs2, 'greater'),   # newer better => diff > 0
    ]:
        mean = diffs.mean()
        half = 1.96 * diffs.std(ddof=1) / (len(diffs) ** 0.5)
        _, p = stats.ttest_1samp(diffs, 0.0, alternative=alt)
        print(f"{label}: mean diff {mean:+.3f} +/- {half:.3f} pts/deal  "
              f"(p={p:.4f} for 'newer is better')")
    both = (stats.ttest_1samp(diffs1, 0.0, alternative='less')[1] < 0.05
            and stats.ttest_1samp(diffs2, 0.0, alternative='greater')[1] < 0.05)
    print(f"Verdict: {'NEWER model is better in BOTH directions (p<0.05)' if both else 'no consistent significant winner'}")

    print(f"\n--- Outcomes (per game, n={tn['games_played']} each) ---")
    print(f"{'Metric':<16} | {'Newer':>10} | {'Older':>10}")
    print("-" * 44)
    for label, key, scale in [
        ("Avg Pts", "total_penalty", None),
        ("Win %", "wins", 100), ("Clean %", "clean", 100),
        ("Disaster %", "disaster", 100), ("QS Eaten %", "qs_eaten", 100),
    ]:
        g_n, g_o = tn["games_played"], to["games_played"]
        vn = tn[key] / g_n * (scale or 1)
        vo = to[key] / g_o * (scale or 1)
        suffix = "%" if scale else " "
        print(f"{label:<16} | {vn:>9.2f}{suffix} | {vo:>9.2f}{suffix}")
    print(f"{'Moons shot':<16} | {tn['moons_shot']:>10} | {to['moons_shot']:>10}")
    print(f"{'Moons conceded':<16} | {tn['moons_conceded']:>10} | {to['moons_conceded']:>10}")

    g_n, g_o = tn["games_played"], to["games_played"]
    print(f"\n--- Points Taken by Phase (avg per game) ---")
    print(f"{'Phase':<16} | {'Newer':>10} | {'Older':>10}")
    print("-" * 44)
    for i, label in enumerate(["Tricks 1-4", "Tricks 5-9", "Tricks 10-13"]):
        print(f"{label:<16} | {tn['phase_pts'][i] / g_n:>10.2f} | {to['phase_pts'][i] / g_o:>10.2f}")

    higher_better = {k: True for k in TACTIC_KEYS}
    higher_better["ak_punished"] = False
    print_rate_table("Tactics (rate [95% Wilson CI], * = significant delta)", [
        ("QS Passed", "qs_pass"),
        ("QS Dump", "qs_dump"), ("Duck Optimal", "duck"), ("Duck (4th seat)", "duck4"),
        ("Slough Quality", "slough"), ("Spade Bleed", "bleed"), ("Lead Safety", "lead_safe"),
        ("Moon Defense", "moon_def"), ("AK-Spade Punish", "ak_punished"),
        ("Trick Control", "control"), ("Forced Win", "forced_win"),
    ], tn, to, higher_better)

    print(f"\n--- Policy Convergence ---")
    print(f"Avg policy entropy (nats): newer {tn['entropy_sum'] / tn['decisions']:.3f} | "
          f"older {to['entropy_sum'] / to['decisions']:.3f}  (lower = more decisive)")

    total_states = sum(div["states"])
    total_dis = sum(div["disagree"])
    print(f"\n--- Policy Divergence (argmax, on {total_states} identical states) ---")
    print(f"Overall disagreement: {total_dis / total_states * 100:.1f}%")
    for i, label in enumerate(["Tricks 1-4", "Tricks 5-9", "Tricks 10-13"]):
        s, dgr = div["states"][i], div["disagree"][i]
        print(f"  {label}: {dgr / s * 100 if s else 0:.1f}%")
    if div["top"]:
        print(f"\nTop {len(div['top'])} disagreement states (by symmetric KL):")
        for kl, _, info in sorted(div["top"], reverse=True):
            print(f"  [KL {kl:6.2f}] {info}")

    print("\n" + "=" * 90)

    # ---------------- CSV history ----------------
    run_time = datetime.datetime.now().isoformat(timespec='seconds')
    rows = []
    for role, name, arch, t, diffs, alt in [
        ("newer", newer_name, newer_arch, tn, diffs1, 'less'),
        ("older", older_name, older_arch, to, diffs2, 'less'),
    ]:
        _, solo_p = stats.ttest_1samp(diffs, 0.0, alternative=alt)
        row = {
            "run_time": run_time, "mode": MODE, "deals": PAIRED_DEALS,
            "model": name, "arch": arch, "role": role,
            "games": t["games_played"],
            "avg_pts": round(t["total_penalty"] / t["games_played"], 4),
            "win_pct": round(t["wins"] / t["games_played"] * 100, 2),
            "clean_pct": round(t["clean"] / t["games_played"] * 100, 2),
            "disaster_pct": round(t["disaster"] / t["games_played"] * 100, 2),
            "qs_eaten_pct": round(t["qs_eaten"] / t["games_played"] * 100, 2),
            "moons_shot": t["moons_shot"], "moons_conceded": t["moons_conceded"],
            "entropy": round(t["entropy_sum"] / t["decisions"], 4),
            "solo_diff_mean": round(float(diffs.mean()), 4),
            "solo_diff_p": round(float(solo_p), 6),
        }
        for k in TACTIC_KEYS:
            n = t[k + "_opps"]
            row[k + "_rate"] = round(t[k + "_succ"] / n * 100, 2) if n else None
            row[k + "_n"] = n
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(HISTORY_CSV, mode='a', index=False, header=not os.path.exists(HISTORY_CSV))
    print(f"Appended results to {HISTORY_CSV}")

if __name__ == '__main__':
    main()
