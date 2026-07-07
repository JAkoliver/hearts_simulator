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

def evaluate_candidate(candidate_path, baseline_path):
    print(f"Evaluating Candidate {candidate_path} vs Baseline {baseline_path}")
    
    candidate_net = HeartsNet()
    candidate_net.load_state_dict(torch.load(candidate_path, weights_only=True))
    candidate_net.eval()
    
    baseline_net = HeartsNet()
    baseline_net.load_state_dict(torch.load(baseline_path, weights_only=True))
    baseline_net.eval()
    
    env = hearts_env.HeartsEnv(seed=int(time.time()))
    
    candidate_scores = []
    baseline_scores = []
    
    num_games = 2500
    
    for game_idx in range(num_games):
        env.reset()
        done = False
        
        # Rotate candidate seating
        candidate_seat = game_idx % 4
        
        while not done:
            obs = env.observe()
            legal_actions_raw = env.get_legal_actions()
            legal_actions = [a for a in legal_actions_raw if a != -1]
            current_player = env.get_current_player()
            
            # Formulate tensor
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.zeros((1, 52), dtype=torch.bool)
            for a in legal_actions:
                mask_tensor[0, a] = True
                
            network = candidate_net if current_player == candidate_seat else baseline_net
            
            with torch.no_grad():
                masked_logits, _ = network(state_tensor, mask_tensor)
                
            # Play deterministically for evaluation: pick max logit
            action = torch.argmax(masked_logits, dim=1).item()
            
            result = env.step(action)
            done = result.done
            
        scores = env.get_round_scores()
        candidate_scores.append(scores[candidate_seat])
        
        for i in range(4):
            if i != candidate_seat:
                baseline_scores.append(scores[i])
                
    cand_mean = np.mean(candidate_scores)
    base_mean = np.mean(baseline_scores)
    print(f"Candidate Mean Score: {cand_mean:.3f}")
    print(f"Baseline Mean Score: {base_mean:.3f}")
    
    # Welch's t-test (one-sided, alternative='less')
    # Because in Hearts, a lower score is better.
    t_stat, p_val = stats.ttest_ind(candidate_scores, baseline_scores, equal_var=False, alternative='less')
    print(f"T-Statistic: {t_stat:.3f}, P-Value: {p_val:.5f}")
    
    is_success = (p_val < 0.05) and (cand_mean < base_mean)
    return is_success, cand_mean

def rollback_config():
    print("Rolling back configuration...")
    if os.path.exists('config_backup.json'):
        shutil.copy('config_backup.json', 'config.json')

def main():
    ledger = get_ledger()
    
    print(f"--- Starting Agentic MLOps Trial ---")
    
    if os.path.exists('hearts_model_final.pth'):
        shutil.copy('hearts_model_final.pth', 'hearts_model_baseline_temp.pth')
        
    try:
        is_valid, err_msg = validate_code()
        
        # Restore baseline immediately to wipe smoke test pollution
        if os.path.exists('hearts_model_baseline_temp.pth'):
            shutil.copy('hearts_model_baseline_temp.pth', 'hearts_model_final.pth')
            
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
        
        success, new_mean = evaluate_candidate(candidate_model_path, baseline_model_path)
        
        with open('config.json', 'r') as f:
            cfg = json.load(f)
            
        if success:
            print(f"*** Experiment SUCCESS! Candidate is statistically superior (mean={new_mean:.3f}) ***")
            
            timestamp = int(time.time())
            milestone_path = f"hearts_model_milestone_{timestamp}.pth"
            shutil.copy('hearts_model_final.pth', milestone_path)
            print(f"*** Elite Milestone Captured: {milestone_path} ***")
            
            ledger["baseline_score"] = new_mean
            write_ledger(ledger)
        else:
            print("Experiment FAILED (Candidate not significantly better). Rolling back.")
            rollback_config()
            shutil.copy('hearts_model_baseline_temp.pth', 'hearts_model_final.pth')
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
        
    finally:
        if os.path.exists('hearts_model_baseline_temp.pth'):
            os.remove('hearts_model_baseline_temp.pth')

if __name__ == '__main__':
    main()
