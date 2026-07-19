import os
import time
import json
import subprocess
import glob
import shutil
import scipy.stats as stats
import torch
import numpy as np

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

def play_round(env, seat_networks):
    """Play one full round deterministically (argmax) and return the 4 scores."""
    env.reset()
    done = False
    while not done:
        obs = env.observe()
        legal_actions_raw = env.get_legal_actions()
        current_player = env.get_current_player()

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
# Search-level promotion gate
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

def _trace_for_search(checkpoint_path, out_path):
    net = net_from_checkpoint(checkpoint_path)
    net.eval()
    dummy_obs = torch.zeros(1, 550, dtype=torch.float32)
    dummy_mask = torch.zeros(1, 52, dtype=torch.bool)
    torch.jit.trace(_SearchExport(net), (dummy_obs, dummy_mask)).save(out_path)

def _search_start(model_pt, opponent_pt, deals, k, seed, out_csv):
    cmd = [SEARCH_EVAL_EXE, '--search-model', model_pt, '--opponent-model', opponent_pt,
           '--deals', str(deals), '--k', str(k), '--pass-search',
           '--seed', str(seed), '--out', out_csv]
    if torch.cuda.is_available():
        cmd.extend(['--cuda', '--bf16'])
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

def _search_finish(proc, out_csv):
    _, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"SearchEval failed (exit {proc.returncode}): {err[-500:]}")
    return np.genfromtxt(out_csv, delimiter=',', names=True)['diff']

# Shard seed stride: matches cloud/gate_shard.py so pairing holds per-row
# whenever both sides of ANY comparison run the same shard layout.
_GATE_SHARD_STRIDE = 1_000_000

def evaluate_candidate_search(candidate_path, deals=500, k=64, alpha=0.05,
                              shards=4):
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

    Returns (success, mean_delta, p)."""
    if not (os.path.exists(SEARCH_EVAL_EXE) and os.path.exists('hearts_ai_search.pt')):
        print("Search gate skipped (SearchEval.exe or hearts_ai_search.pt missing).")
        return True, None, None
    if os.path.exists(NEUTRAL_OPPONENT):
        opponent = NEUTRAL_OPPONENT
    else:
        print("WARNING: neutral anchor missing; falling back to the baseline's own "
              "raw trace as opponents (biased toward the baseline).")
        opponent = 'hearts_ai_grandmaster.pt'

    seed = int(time.time())
    _trace_for_search(candidate_path, 'search_gate_candidate.pt')
    shards = max(1, min(shards, deals))
    print(f"Search gate: {deals} paired deals, K={k}, {shards} shard pairs, "
          f"neutral opponents (all {2 * shards} runs concurrent)...")
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
                      _search_start('search_gate_candidate.pt', opponent, n, k, s, c_csv),
                      c_csv,
                      _search_start('hearts_ai_search.pt', opponent, n, k, s, b_csv),
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
    return is_success, mean, float(p_val)

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

        # Gate design (2026-07-15): raw play and searched strength can point
        # in opposite directions, and the deployed player is a SEARCH player.
        # The raw evaluation is only a sanity guard (reject candidates that
        # are substantially worse raw); promotion is decided by the search
        # gate against neutral opponents.
        fail_reason = "evaluation_failed"
        if os.path.exists(baseline_model_path):
            raw_sig, new_mean, raw_diff = evaluate_candidate(candidate_model_path, baseline_model_path)
            guard = cfg.get('raw_guard_threshold', 0.3)
            if raw_diff > guard:
                print(f"Raw guard FAILED: candidate {raw_diff:+.3f} vs baseline "
                      f"(threshold +{guard}). Skipping search gate.")
                success = False
                fail_reason = "raw_guard_failed"
            else:
                print(f"Raw guard passed ({raw_diff:+.3f} <= +{guard}"
                      f"{', significantly better' if raw_sig else ''}). Search gate decides.")
                success, sg_mean, sg_p = evaluate_candidate_search(
                    candidate_model_path,
                    deals=cfg.get('search_gate_deals', 800),
                    k=cfg.get('search_gate_k', 64),
                    alpha=cfg.get('search_gate_alpha', 0.02))
                if not success:
                    fail_reason = "search_gate_failed"
        else:
            print("No baseline exists (fresh start). Auto-promoting first candidate as the initial baseline.")
            success, new_mean = True, None

        if success:
            mean_str = f"{new_mean:.3f}" if new_mean is not None else "bootstrap"
            print(f"*** Experiment SUCCESS! Candidate is statistically superior (mean={mean_str}) ***")
            
            timestamp = int(time.time())
            os.makedirs('Hall_of_Fame', exist_ok=True)
            milestone_path = f"Hall_of_Fame/hearts_model_milestone_{timestamp}.pth"
            shutil.copy('hearts_model_final.pth', milestone_path)
            print(f"*** Elite Milestone Captured: {milestone_path} ***")
            
            ledger["baseline_score"] = new_mean
            write_ledger(ledger)
            
            print("Automating C++ deployment... Tracing network architecture...")
            export_res = subprocess.run(["python", "export.py"])
            if export_res.returncode == 0:
                print("Deployment asset 'hearts_ai_grandmaster.pt' successfully updated!")
            else:
                print("Failed to export deployment asset.")
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
