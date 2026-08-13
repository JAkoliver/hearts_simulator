import os
import time
import json
import subprocess
import glob
import shutil
import scipy.stats as stats
import torch
import numpy as np

import headroom

from hearts_net import HeartsNet, net_from_checkpoint
import hearts_env

LEDGER_FILE = 'experiment_ledger.json'

def get_ledger():
    with open(LEDGER_FILE, 'r') as f:
        return json.load(f)

def write_ledger(ledger):
    with open(LEDGER_FILE, 'w') as f:
        json.dump(ledger, f, indent=4)

def validate_code():
    print("Running syntax check...")
    res = subprocess.run(["python", "-m", "py_compile", "train.py"], capture_output=True, text=True)
    if res.returncode != 0:
        return False, f"Syntax error:\n{res.stderr}"
        
    print("Running smoke test (SMOKE_TEST=1)...")
    env_vars = os.environ.copy()
    env_vars["SMOKE_TEST"] = "1"
    res = subprocess.run(["python", "train.py"], env=env_vars, capture_output=True, text=True)
    if res.returncode != 0:
        return False, f"Smoke test failed:\n{res.stderr}\n{res.stdout}"
    return True, ""

def _obs_for_net(env, net):
    """Per-net observation: v6 nets (obs_dim 882) get obs v2 assembled as
    [observe()[0:550] | zero match ctx | observe_ext()]; zero match
    context is the start-of-match state, the same thing the
    match-conditioned v5 baseline sees in this per-deal instrument
    (its 550-dim input skips the match term entirely). Everything else
    gets the classic observe() untouched - bit-identical legacy path."""
    obs = env.observe()
    if getattr(net, 'obs_dim', 0) == 882:
        return np.concatenate([
            np.asarray(obs, dtype=np.float32)[:550],
            np.zeros(6, dtype=np.float32),
            np.asarray(env.observe_ext(), dtype=np.float32)])
    return obs


def play_round(env, seat_networks):
    """Play one full round deterministically (argmax) and return the 4 scores."""
    env.reset()
    done = False
    while not done:
        current_player = env.get_current_player()
        obs = _obs_for_net(env, seat_networks[current_player])
        legal_actions_raw = env.get_legal_actions()

        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.zeros((1, 52), dtype=torch.bool)
        for a in legal_actions_raw:
            if a != -1:
                mask_tensor[0, a] = True

        with torch.no_grad():
            masked_logits, _ = seat_networks[current_player](state_tensor, mask_tensor)

        action = torch.argmax(masked_logits, dim=1).item()
        result = env.step(action)
        done = result.done

    return env.get_round_scores()

def _eval_chunk(job):
    """One worker's share of the paired-deal evaluation (own env pair, own
    seed). Runs in a multiprocessing worker; must stay module-level
    picklable."""
    candidate_path, baseline_path, seed, deal_offset, n_deals = job
    torch.set_num_threads(1)  # N workers x default threads would thrash

    # net_from_checkpoint infers width/depth, so candidate and baseline may be
    # different architectures (e.g. capacity-test candidates)
    candidate_net = net_from_checkpoint(candidate_path)
    candidate_net.eval()

    baseline_net = net_from_checkpoint(baseline_path)
    baseline_net.eval()

    env_cand = hearts_env.HeartsEnv(seed=seed)
    env_base = hearts_env.HeartsEnv(seed=seed)

    diffs = []
    candidate_scores = []
    for i in range(n_deals):
        # Global deal index keeps the 4-seat rotation balanced across chunks
        candidate_seat = (deal_offset + i) % 4

        seats = [baseline_net] * 4
        seats[candidate_seat] = candidate_net
        cand_table_scores = play_round(env_cand, seats)
        base_table_scores = play_round(env_base, [baseline_net] * 4)

        candidate_scores.append(cand_table_scores[candidate_seat])
        diffs.append(cand_table_scores[candidate_seat] - base_table_scores[candidate_seat])
    return diffs, candidate_scores

def evaluate_candidate(candidate_path, baseline_path, num_deals=2500, workers=8):
    print(f"Evaluating Candidate {candidate_path} vs Baseline {baseline_path}")

    # Paired duplicate deals: the engine's RNG is consumed only by reset()'s
    # shuffle, so two envs built from the same seed produce identical deal
    # sequences regardless of play. Each deal is played twice — once with the
    # candidate seated, once all-baseline — and we test the per-deal score
    # differential at the same seat. This removes deal/seat luck, the dominant
    # variance source, and keeps the samples statistically independent.
    #
    # The deals are split across worker processes, each with its own env pair
    # on its own seed — chunks are just independent batches of paired samples,
    # so the pooled t-test is unchanged.
    seed = int(time.time())
    base = num_deals // workers
    jobs = []
    offset = 0
    for w in range(workers):
        n = base + (1 if w < num_deals % workers else 0)
        if n == 0:
            continue
        jobs.append((candidate_path, baseline_path, seed + w, offset, n))
        offset += n

    import multiprocessing
    with multiprocessing.Pool(len(jobs)) as pool:
        results = pool.map(_eval_chunk, jobs)

    diffs = []
    candidate_scores = []
    for d, c in results:
        diffs.extend(d)
        candidate_scores.extend(c)

    diffs = np.array(diffs, dtype=np.float64)
    cand_mean = np.mean(candidate_scores)
    mean_diff = diffs.mean()
    print(f"Candidate Mean Score: {cand_mean:.3f}")
    print(f"Mean Paired Differential (negative = candidate better): {mean_diff:+.3f}")

    # One-sample t-test on paired differentials (one-sided, alternative='less')
    # Because in Hearts, a lower score is better. Alpha is 0.01 rather than
    # 0.05 because this gate runs repeatedly: at 0.05, one in twenty no-better
    # candidates would be promoted by luck.
    t_stat, p_val = stats.ttest_1samp(diffs, 0.0, alternative='less')
    print(f"T-Statistic: {t_stat:.3f}, P-Value: {p_val:.5f}")

    is_success = bool(p_val < 0.01) and (mean_diff < 0)
    return is_success, cand_mean, float(mean_diff)

# ---------------------------------------------------------------------------
# Neutral-opponent raw promotion gate (raw-line promotion, 2026-07-21)
# ---------------------------------------------------------------------------
# The head-to-head raw evaluation above seats the candidate among the very
# baseline it trained against; measured 2026-07-21, it is biased in BOTH
# directions (a candidate whose head-to-head read -0.187 ns was truly -0.636
# vs neutral tables). The promoter therefore seats candidate and baseline,
# one at a time, at the same seat of identical deals against three neutral
# v3-m7 anchor seats and tests the paired per-deal differential. Rationale
# and the A/B diagnostics behind this design: docs/ppo_v5_round2_findings.md.

class _LegacySeat(torch.nn.Module):
    """Adapts the 238-dim v3 anchor trace to the current 550-dim observation
    (prefix layout, same convention SearchPlayer.hpp probes for)."""

    def __init__(self, traced):
        super().__init__()
        self.traced = traced

    def forward(self, observation, legal_actions_mask):
        return self.traced(observation[:, :238], legal_actions_mask)


def _neutral_chunk(job):
    candidate_path, baseline_path, anchor_path, seed, deal_offset, n_deals = job
    torch.set_num_threads(1)

    cand = net_from_checkpoint(candidate_path)
    cand.eval()
    base = net_from_checkpoint(baseline_path)
    base.eval()
    neutral = _LegacySeat(torch.jit.load(anchor_path))
    neutral.eval()

    # Same-seed envs stay deal-synchronized (RNG consumed only by reset()).
    env_a = hearts_env.HeartsEnv(seed=seed)
    env_b = hearts_env.HeartsEnv(seed=seed)

    diffs = []
    for i in range(n_deals):
        seat = (deal_offset + i) % 4
        seats_a = [neutral] * 4
        seats_a[seat] = cand
        a_scores = play_round(env_a, seats_a)
        seats_b = [neutral] * 4
        seats_b[seat] = base
        b_scores = play_round(env_b, seats_b)
        diffs.append(a_scores[seat] - b_scores[seat])
    return diffs


def evaluate_candidate_neutral_raw(candidate_path, baseline_path,
                                   num_deals=2500, workers=12, alpha=0.05):
    """Raw-line promoter. Returns (success, mean, se, p)."""
    workers = headroom.scaled_workers(workers)
    anchor = NEUTRAL_OPPONENT
    if not os.path.exists(anchor):
        raise RuntimeError(f"neutral anchor missing ({anchor}); "
                           "cannot run the raw promotion gate")
    seed = int(time.time())
    print(f"Neutral raw gate (promoter): {num_deals} paired deals, "
          f"candidate/baseline @ seat vs 3x v3-m7 anchors, seed {seed}")

    per = num_deals // workers
    extra = num_deals % workers
    jobs, offset = [], 0
    for w in range(workers):
        n = per + (1 if w < extra else 0)
        if n == 0:
            continue
        jobs.append((candidate_path, baseline_path, anchor,
                     seed + w * _GATE_SHARD_STRIDE, offset, n))
        offset += n

    import multiprocessing
    with multiprocessing.Pool(len(jobs),
                              initializer=headroom.apply_process_priority) as pool:
        results = pool.map(_neutral_chunk, jobs)

    diffs = np.array([d for r in results for d in r], dtype=np.float64)
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    t_stat, p_val = stats.ttest_1samp(diffs, 0.0, alternative='less')
    print(f"Neutral raw delta (negative = candidate better): {mean:+.3f} "
          f"(SE {se:.3f}, n={len(diffs)})")
    print(f"T-Statistic: {t_stat:.3f}, P-Value: {p_val:.5f}")
    success = bool(p_val < alpha) and (mean < 0)
    return success, mean, se, float(p_val)

def evaluate_candidate_match(candidate_path, baseline_path,
                             matches=800, workers=12, alpha=0.05):
    """Match-to-100 promoter (docs/ROADMAP.md phase 1): paired matches vs
    neutral anchors via match_eval.run_gate. Promotes on significantly
    better mean placement. Returns (success, dplace_mean, dplace_se, p)."""
    import match_eval  # deferred: match_eval imports from this module
    r = match_eval.run_gate(candidate_path, baseline_path,
                            matches=matches, workers=workers)
    success = bool(r['p_place'] < alpha) and (r['dplace_mean'] < 0)
    return success, r['dplace_mean'], r['dplace_se'], r['p_place']

# ---------------------------------------------------------------------------
# Search-level gate (promoter until 2026-07-21; non-regression guard since)
# ---------------------------------------------------------------------------
# Raw head-to-head strength and SEARCHED strength can point in opposite
# directions (measured 2026-07-14: a candidate -0.39 better raw was +1.24
# worse as a search substrate). The deployed player and the expert-iteration
# teacher are search players, so promotion additionally requires the
# candidate to be at least as strong UNDER SEARCH. Opponents must be NEUTRAL
# (a third-party anchor): evaluating against the baseline's own raw policy
# hands the baseline a perfect-opponent-model advantage no candidate can
# match.

NEUTRAL_OPPONENT = os.path.join('legacy_v3_pass238', 'hearts_ai_grandmaster_v3_milestone7.pt')
SEARCH_EVAL_EXE = os.path.join('build', 'Release', 'SearchEval.exe')

class _SearchExport(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, observation, legal_actions_mask):
        return self.net.forward_all(observation, legal_actions_mask)

def _trace_for_search(checkpoint_path, out_path, obs_dim=550):
    net = net_from_checkpoint(checkpoint_path)
    net.eval()
    # v6 (obs-v2) nets accept ONLY their own width - tracing one at 550/556
    # raises inside the net. The engine probes 882 and assembles the
    # matching row (SearchPlayer::FillObsRow), so the NET decides, not the
    # caller's match-aware guess.
    net_dim = int(getattr(net, 'obs_dim', 0) or 0)
    if net_dim == 882 and obs_dim != 882:
        print(f"  (obs-v2 net: tracing at 882, not {obs_dim})")
        obs_dim = 882
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    dummy_mask = torch.zeros(1, 52, dtype=torch.bool)
    torch.jit.trace(_SearchExport(net), (dummy_obs, dummy_mask)).save(out_path)

# Guard evolution (rules #16, validated 2026-07-27): the deployed ceiling is
# MATCH-AWARE search (556 ctx + equity leaves). When the equity trace exists,
# the guard runs BOTH arms match-aware so it protects the actual substrate.
# Single-deal context means K stays at the base value (no >=85 totals), so
# the endgame path rides on the separately-validated K schedule.
EQUITY_TRACE = 'hearts_equity.pt'
MATCH_SEARCH_TRACE = 'hearts_ai_search_match.pt'

def _guard_match_aware():
    return os.path.exists(EQUITY_TRACE) and os.path.exists(MATCH_SEARCH_TRACE)

def _search_start(model_pt, opponent_pt, deals, k, seed, out_csv,
                  equity_pt=None):
    cmd = [SEARCH_EVAL_EXE, '--search-model', model_pt, '--opponent-model', opponent_pt,
           '--deals', str(deals), '--k', str(k), '--pass-search',
           '--seed', str(seed), '--out', out_csv]
    if equity_pt:
        cmd.extend(['--equity-model', equity_pt])
    if torch.cuda.is_available():
        cmd.extend(['--cuda', '--bf16'])
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    headroom.child_priority(proc.pid)
    return proc

def _search_finish(proc, out_csv):
    _, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"SearchEval failed (exit {proc.returncode}): {err[-500:]}")
    return np.genfromtxt(out_csv, delimiter=',', names=True)['diff']

# Shard seed stride: matches cloud/gate_shard.py so pairing holds per-row
# whenever both sides of ANY comparison run the same shard layout.
_GATE_SHARD_STRIDE = 1_000_000

def evaluate_candidate_search(candidate_path, deals=500, k=64, alpha=0.05,
                              shards=4, baseline_ckpt=None):
    """Paired search-vs-search gate: candidate and the current teacher trace
    each play `deals` identical deals (1 search seat vs 3 neutral-anchor raw
    seats, table B all-anchor), one-sided t-test on the per-deal delta.

    Sharded (2*shards concurrent SearchEval processes): the gate was
    re-powered 2026-07-19 from n=600 to n=2400 because at per-deal std ~7.6
    the old gate only promoted effects stronger than -0.51/deal (~25% power
    against a true -0.3) - early-era promotions were -0.7..-1.2 so it never
    mattered, but mature gains are -0.1..-0.3 and were being discarded.
    n=2400 puts the promotion bar near -0.26 at the same alpha; sharding
    keeps wall-clock manageable (~1.2x single-process rate on the 4090,
    ~2x on high-core cloud GPUs).

    Since 2026-07-21 this gate no longer promotes: the raw-line promoter
    (evaluate_candidate_neutral_raw) decides, and main() uses this gate's
    mean/se as a non-regression guard on the deployed search player.

    Returns (success, mean_delta, p, se)."""
    if not (os.path.exists(SEARCH_EVAL_EXE) and os.path.exists('hearts_ai_search.pt')):
        print("Search gate skipped (SearchEval.exe or hearts_ai_search.pt missing).")
        return True, None, None, None
    if os.path.exists(NEUTRAL_OPPONENT):
        opponent = NEUTRAL_OPPONENT
    else:
        print("WARNING: neutral anchor missing; falling back to the baseline's own "
              "raw trace as opponents (biased toward the baseline).")
        opponent = 'hearts_ai_grandmaster.pt'

    match_aware = _guard_match_aware()
    equity = EQUITY_TRACE if match_aware else None
    baseline_trace = MATCH_SEARCH_TRACE if match_aware else 'hearts_ai_search.pt'
    if baseline_ckpt:
        # CANDIDATE-LINEAGE guard (v6 prereg stage 4): compare the candidate
        # against THIS lineage's own previous champion, not the deployed
        # teacher trace. A separate lineage that has not yet caught the
        # champion would fail a champion-anchored guard on every trial, so
        # the ladder could never climb - the same conflation that made the
        # first two manual trials uninterpretable (ledger 2026-08-13).
        _trace_for_search(baseline_ckpt, 'search_gate_baseline.pt',
                          obs_dim=556 if match_aware else 550)
        baseline_trace = 'search_gate_baseline.pt'
        print(f"  guard baseline: {baseline_ckpt} (lineage's own)")
    seed = int(time.time())
    _trace_for_search(candidate_path, 'search_gate_candidate.pt',
                      obs_dim=556 if match_aware else 550)
    shards = headroom.scaled_shards(max(1, min(shards, deals)))
    print(f"Search gate: {deals} paired deals, K={k}, {shards} shard pairs, "
          f"neutral opponents, "
          f"{'MATCH-AWARE (equity leaves, rules #16)' if match_aware else 'match-blind'} "
          f"(all {2 * shards} runs concurrent)...")
    per = deals // shards
    extra = deals % shards
    procs = []  # (shard_idx, n, cand_proc, cand_csv, base_proc, base_csv)
    for i in range(shards):
        n = per + (1 if i < extra else 0)
        if n == 0:
            continue
        s = seed + i * _GATE_SHARD_STRIDE
        c_csv = f'search_eval_gate_cand_s{i}.csv'
        b_csv = f'search_eval_gate_base_s{i}.csv'
        procs.append((i, n,
                      _search_start('search_gate_candidate.pt', opponent, n, k, s, c_csv,
                                    equity_pt=equity),
                      c_csv,
                      _search_start(baseline_trace, opponent, n, k, s, b_csv,
                                    equity_pt=equity),
                      b_csv))
    cand_parts, base_parts = [], []
    for i, n, pc, c_csv, pb, b_csv in procs:
        c = _search_finish(pc, c_csv)
        b = _search_finish(pb, b_csv)
        if len(np.atleast_1d(c)) != n or len(np.atleast_1d(b)) != n:
            raise RuntimeError(f"gate shard {i}: row count mismatch "
                               f"({len(np.atleast_1d(c))}/{len(np.atleast_1d(b))} vs {n})")
        cand_parts.append(np.atleast_1d(c))
        base_parts.append(np.atleast_1d(b))
    cand = np.concatenate(cand_parts)
    base = np.concatenate(base_parts)
    delta = cand - base
    t_stat, p_val = stats.ttest_1samp(delta, 0.0, alternative='less')
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    print(f"Search gate paired delta (negative = candidate better): {mean:+.3f} "
          f"(SE {se:.3f}, n={len(delta)})")
    print(f"T-Statistic: {t_stat:.3f}, P-Value: {p_val:.5f}")
    is_success = bool(p_val < alpha) and (mean < 0)
    return is_success, mean, float(p_val), se

def rollback_config():
    print("Rolling back configuration...")
    if os.path.exists('config_backup.json'):
        shutil.copy('config_backup.json', 'config.json')

def restore_optimizer_state():
    # Keep hearts_optimizer.pth consistent with whichever model file we just
    # restored: put back the baseline's optimizer state if we had one, or
    # remove the candidate's orphaned state if we didn't
    if os.path.exists('hearts_optimizer_baseline_temp.pth'):
        shutil.copy('hearts_optimizer_baseline_temp.pth', 'hearts_optimizer.pth')
    elif os.path.exists('hearts_optimizer.pth'):
        os.remove('hearts_optimizer.pth')

def main():
    ledger = get_ledger()
    
    print(f"--- Starting Agentic MLOps Trial ---")
    
    if os.path.exists('hearts_model_final.pth'):
        shutil.copy('hearts_model_final.pth', 'hearts_model_baseline_temp.pth')
    if os.path.exists('hearts_optimizer.pth'):
        shutil.copy('hearts_optimizer.pth', 'hearts_optimizer_baseline_temp.pth')

    try:
        is_valid, err_msg = validate_code()

        # Restore baseline immediately to wipe smoke test pollution. On a fresh
        # start (no baseline existed), the smoke test's output IS pollution, so
        # remove it and let the real run begin from scratch.
        if os.path.exists('hearts_model_baseline_temp.pth'):
            shutil.copy('hearts_model_baseline_temp.pth', 'hearts_model_final.pth')
        elif os.path.exists('hearts_model_final.pth'):
            os.remove('hearts_model_final.pth')
        restore_optimizer_state()
            
        if not is_valid:
            print("Validation Sandbox Failed!")
            print(err_msg)
            rollback_config()
            
            with open('config.json', 'r') as f:
                cfg = json.load(f)
            ledger["failed_experiments"].append({
                "config": cfg,
                "reason": "validation_failed"
            })
            write_ledger(ledger)
            return
            
        print("Validation passed. Launching full training loop...")
        train_res = subprocess.run(["python", "train.py"])
        if train_res.returncode != 0:
            print("Training crashed or was interrupted!")
            rollback_config()
            restore_optimizer_state()
            with open('config.json', 'r') as f:
                cfg = json.load(f)
            ledger["failed_experiments"].append({
                "config": cfg,
                "reason": "training_crashed"
            })
            write_ledger(ledger)
            return
            
        print("Training Complete. Moving to Head-to-Head Evaluator...")
        candidate_model_path = 'hearts_model_final.pth'
        baseline_model_path = 'hearts_model_baseline_temp.pth'

        # Training can exit 0 without producing a model (e.g. killed externally
        # after the KeyboardInterrupt handler ran). Don't evaluate a ghost.
        if not os.path.exists(candidate_model_path):
            print("No candidate model was produced — treating trial as crashed.")
            rollback_config()
            restore_optimizer_state()
            with open('config.json', 'r') as f:
                cfg = json.load(f)
            ledger["failed_experiments"].append({
                "config": cfg,
                "reason": "no_candidate_produced"
            })
            write_ledger(ledger)
            return

        with open('config.json', 'r') as f:
            cfg = json.load(f)

        # Gate design (2026-07-21, raw-line promotion): the project goal is a
        # RAW net at best-human level, and diagnostics A/B showed PPO's raw
        # gains are genuine vs neutral opponents while searched strength is
        # saturated from this baseline (docs/ppo_v5_round2_findings.md).
        # Promotion is decided by the neutral raw gate; the search gate
        # remains only as a non-regression guard so the deployed search
        # player never gets measurably worse.
        fail_reason = "evaluation_failed"
        if os.path.exists(baseline_model_path):
            if cfg.get('match_mode', False):
                # Match-era promoter (2026-07-23): placement across paired
                # matches decides; the neutral raw gate is demoted to an
                # informational telemetry line (experiment_rules.md #5).
                promo_success, new_mean, promo_se, promo_p = evaluate_candidate_match(
                    candidate_model_path, baseline_model_path,
                    matches=cfg.get('match_gate_matches', 800),
                    workers=cfg.get('raw_gate_workers', 12),
                    alpha=cfg.get('match_gate_alpha', 0.05))
                _, info_raw, info_se, info_p = evaluate_candidate_neutral_raw(
                    candidate_model_path, baseline_model_path,
                    num_deals=cfg.get('raw_gate_deals', 2500),
                    workers=cfg.get('raw_gate_workers', 12),
                    alpha=cfg.get('raw_gate_alpha', 0.05))
                print(f"[telemetry] neutral raw delta {info_raw:+.3f} "
                      f"(SE {info_se:.3f}, p={info_p:.3f}) - informational only")
                gate_name = "Match gate"
            else:
                promo_success, new_mean, promo_se, promo_p = evaluate_candidate_neutral_raw(
                    candidate_model_path, baseline_model_path,
                    num_deals=cfg.get('raw_gate_deals', 2500),
                    workers=cfg.get('raw_gate_workers', 12),
                    alpha=cfg.get('raw_gate_alpha', 0.05))
                gate_name = "Neutral raw gate"
            if not promo_success:
                print(f"{gate_name} FAILED ({new_mean:+.3f}, p={promo_p:.3f}). "
                      "Skipping search guard.")
                success = False
                fail_reason = gate_name.lower().replace(' ', '_') + "_failed"
            else:
                print(f"{gate_name} PASSED ({new_mean:+.3f}, p={promo_p:.5f}). "
                      "Search non-regression guard decides.")
                _, sg_mean, sg_p, sg_se = evaluate_candidate_search(
                    candidate_model_path,
                    deals=cfg.get('search_gate_deals', 2400),
                    k=cfg.get('search_gate_k', 32),
                    alpha=cfg.get('search_gate_alpha', 0.05),
                    baseline_ckpt=(baseline_model_path
                                   if cfg.get('candidate_lineage') else None))
                if sg_mean is None:
                    # Guard infrastructure missing; evaluate_candidate_search
                    # already printed the skip. Do not promote blind.
                    success = False
                    fail_reason = "search_guard_unavailable"
                else:
                    margin = cfg.get('search_guard_margin', 0.3)
                    ub = sg_mean + float(stats.t.ppf(0.95,
                        cfg.get('search_gate_deals', 2400) - 1)) * sg_se
                    success = ub <= margin
                    print(f"Search non-regression guard: delta {sg_mean:+.3f} "
                          f"(SE {sg_se:.3f}), one-sided 95% UB {ub:+.3f} vs "
                          f"margin +{margin} -> {'PASS' if success else 'FAIL'}")
                    if not success:
                        fail_reason = "search_regression"
        else:
            print("No baseline exists (fresh start). Auto-promoting first candidate as the initial baseline.")
            success, new_mean = True, None

        if success:
            mean_str = f"{new_mean:.3f}" if new_mean is not None else "bootstrap"
            print(f"*** Experiment SUCCESS! Candidate is statistically superior (mean={mean_str}) ***")
            
            timestamp = int(time.time())
            # CANDIDATE LINEAGE: milestones stay OUT of Hall_of_Fame - that
            # directory is the champion's archive AND train.py's PPO opponent
            # pool, so a not-yet-promoted lineage must not seed either.
            lineage = bool(cfg.get('candidate_lineage'))
            mdir = cfg.get('lineage_dir', 'v6_stage4/milestones') if lineage                 else 'Hall_of_Fame'
            os.makedirs(mdir, exist_ok=True)
            milestone_path = f"{mdir}/hearts_model_milestone_{timestamp}.pth"
            shutil.copy('hearts_model_final.pth', milestone_path)
            print(f"*** Elite Milestone Captured: {milestone_path} ***")
            
            ledger["baseline_score"] = new_mean
            write_ledger(ledger)
            
            if lineage:
                # A lineage champion is NOT the deployed champion: leave the
                # webapp weights, hearts_ai_grandmaster.pt and the match
                # traces alone. hearts_ai_search_match.pt in particular is
                # the registered v6 teacher, a released artifact, AND the
                # guard baseline for the champion line - overwriting it from
                # a candidate lineage would corrupt all three.
                print("Candidate lineage: deployment artifacts NOT refreshed "
                      "(webapp weights and champion traces untouched).")
                write_ledger(ledger)
                return
            # The webapp serves ONLY promoted weights (hearts_web_model.pth);
            # the chain's hearts_model_final.pth holds unpromoted candidates
            # mid-trial and must never reach the site.
            shutil.copy('hearts_model_final.pth', 'hearts_web_model.pth')
            print("Webapp deployment weights refreshed (hearts_web_model.pth).")
            print("Automating C++ deployment... Tracing network architecture...")
            export_res = subprocess.run(["python", "export.py"])
            if export_res.returncode == 0:
                print("Deployment asset 'hearts_ai_grandmaster.pt' successfully updated!")
            else:
                print("Failed to export deployment asset.")
            # Rules #16: the guard baseline is the MATCH-AWARE trace family -
            # refresh it too so the next trial guards against the new champion.
            if os.path.exists('export_match.py'):
                em = subprocess.run(["python", "export_match.py"])
                print("Match-aware traces re-exported."
                      if em.returncode == 0 else
                      "WARNING: export_match.py failed - guard baseline is STALE.")
        else:
            print("Experiment FAILED (Candidate not significantly better). Rolling back.")
            # Keep the rejected weights for post-hoc analysis (e.g. checking
            # a raw-gate reject's SEARCHED strength) - rollback used to
            # discard them irretrievably
            shutil.copy(candidate_model_path, 'hearts_model_last_rejected.pth')
            rollback_config()
            shutil.copy('hearts_model_baseline_temp.pth', 'hearts_model_final.pth')
            restore_optimizer_state()
            ledger["failed_experiments"].append({
                "config": cfg,
                "reason": fail_reason
            })
            write_ledger(ledger)
            
    except KeyboardInterrupt:
        print("\nOrchestrator interrupted by user.")
        rollback_config()
        if os.path.exists('hearts_model_baseline_temp.pth'):
            shutil.copy('hearts_model_baseline_temp.pth', 'hearts_model_final.pth')
        restore_optimizer_state()

    finally:
        if os.path.exists('hearts_model_baseline_temp.pth'):
            os.remove('hearts_model_baseline_temp.pth')
        if os.path.exists('hearts_optimizer_baseline_temp.pth'):
            os.remove('hearts_optimizer_baseline_temp.pth')

if __name__ == '__main__':
    main()
