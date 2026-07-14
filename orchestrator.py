import os
import time
import json
import subprocess
import glob
import shutil
import scipy.stats as stats
import torch
import numpy as np

from hearts_net import HeartsNet
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

    candidate_net = HeartsNet()
    candidate_net.load_state_dict(torch.load(candidate_path, weights_only=True))
    candidate_net.eval()

    baseline_net = HeartsNet()
    baseline_net.load_state_dict(torch.load(baseline_path, weights_only=True))
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
    return is_success, cand_mean

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

        if os.path.exists(baseline_model_path):
            success, new_mean = evaluate_candidate(candidate_model_path, baseline_model_path)
        else:
            print("No baseline exists (fresh start). Auto-promoting first candidate as the initial baseline.")
            success, new_mean = True, None
        
        with open('config.json', 'r') as f:
            cfg = json.load(f)
            
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
            rollback_config()
            shutil.copy('hearts_model_baseline_temp.pth', 'hearts_model_final.pth')
            restore_optimizer_state()
            ledger["failed_experiments"].append({
                "config": cfg,
                "reason": "evaluation_failed"
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
