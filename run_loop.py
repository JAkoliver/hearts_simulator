import os
import json
import math
import random
import time
import subprocess
import copy
import shutil

CONFIG_FILE = 'config.json'
LEDGER_FILE = 'experiment_ledger.json'

def load_config():
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def load_failed_experiments():
    if not os.path.exists(LEDGER_FILE):
        return []
    with open(LEDGER_FILE, 'r') as f:
        ledger = json.load(f)
    return [exp.get("config") for exp in ledger.get("failed_experiments", [])]

def mutate_config(base_config):
    new_config = copy.deepcopy(base_config)
    
    # 30% chance to mutate each parameter
    mutated = False
    
    # Ensure at least one mutation happens if all rolls fail
    if random.random() > 0.5:
        force_mutate = random.choice(['learning_rate', 'update_timestep', 'gamma',
                                      'eps_clip', 'entropy_coef', 'gae_lambda', 'aux_coef'])
    else:
        force_mutate = None

    if random.random() < 0.3 or force_mutate == 'learning_rate':
        multiplier = random.uniform(0.7, 1.3)
        new_config['learning_rate'] = round(new_config['learning_rate'] * multiplier, 6)
        mutated = True
        
    if random.random() < 0.3 or force_mutate == 'update_timestep':
        delta = random.randint(-15, 15)
        new_config['games_per_worker'] = max(5, new_config['games_per_worker'] + delta)
        new_config['update_timestep'] = new_config['games_per_worker'] * new_config['max_workers']
        mutated = True
        
    if random.random() < 0.3 or force_mutate == 'gamma':
        # Gamma changes the objective, not just the optimizer. With 16-step
        # episodes (3 pass + 13 play) and terminal-only reward, gamma 0.92
        # cuts the credit reaching the passing decisions to ~a third — it
        # starves exactly the pass-phase skills. Keep the walk pinned near 1.0.
        delta = random.uniform(-0.01, 0.01)
        new_config['gamma'] = round(min(1.0, max(0.97, new_config['gamma'] + delta)), 6)
        mutated = True

    if random.random() < 0.3 or force_mutate == 'eps_clip':
        # Bounded above: beyond ~0.3 the PPO trust region is effectively gone
        delta = random.uniform(-0.05, 0.05)
        new_config['eps_clip'] = round(min(0.3, max(0.05, new_config['eps_clip'] + delta)), 6)
        mutated = True

    if random.random() < 0.3 or force_mutate == 'entropy_coef':
        multiplier = random.uniform(0.7, 1.3)
        new_config['entropy_coef'] = round(min(0.05, max(0.001, new_config.get('entropy_coef', 0.01) * multiplier)), 6)
        mutated = True

    if random.random() < 0.3 or force_mutate == 'gae_lambda':
        delta = random.uniform(-0.03, 0.03)
        new_config['gae_lambda'] = round(min(1.0, max(0.8, new_config.get('gae_lambda', 0.95) + delta)), 6)
        mutated = True

    if random.random() < 0.3 or force_mutate == 'aux_coef':
        multiplier = random.uniform(0.7, 1.3)
        new_config['aux_coef'] = round(min(1.0, max(0.1, new_config.get('aux_coef', 0.5) * multiplier)), 6)
        mutated = True

    # Safety net: If by some tiny probability nothing was mutated and force_mutate failed to trigger
    if not mutated:
        new_config['eps_clip'] = round(min(0.3, max(0.05, new_config['eps_clip'] + random.uniform(-0.01, 0.01))), 6)

    return new_config

def configs_match(cfg_a, cfg_b, rel_tol=0.05):
    # Exact float equality never fires against random.uniform products, so
    # treat configs within 5% on every numeric parameter as the same experiment
    if cfg_a is None or cfg_b is None:
        return False
    if set(cfg_a.keys()) != set(cfg_b.keys()):
        return False
    for key, val_a in cfg_a.items():
        val_b = cfg_b[key]
        if isinstance(val_a, float) or isinstance(val_b, float):
            if not math.isclose(val_a, val_b, rel_tol=rel_tol, abs_tol=1e-9):
                return False
        elif val_a != val_b:
            return False
    return True

def is_config_failed(new_config, failed_configs):
    for failed_cfg in failed_configs:
        if configs_match(new_config, failed_cfg):
            return True
    return False

def generate_safe_mutation(base_config, failed_configs):
    attempts = 0
    while True:
        attempts += 1
        new_config = mutate_config(base_config)
        if not is_config_failed(new_config, failed_configs):
            print(f"Generated safe mutation after {attempts} attempts.")
            return new_config
        
        # Anti-infinite loop protection
        if attempts > 1000:
            print("Warning: Could not find a novel configuration after 1000 attempts.")
            return new_config

def main():
    print("Initializing Autonomous Experimentation Loop...")
    while True:
        # 1. Read current state
        base_config = load_config()
        failed_configs = load_failed_experiments()
        
        # Backup the known-good baseline config
        shutil.copy('config.json', 'config_backup.json')
        
        # 2. Generate a novel mutation that hasn't failed before
        new_config = generate_safe_mutation(base_config, failed_configs)
        
        # 3. Save the mutated configuration
        save_config(new_config)
        print("\n--- Next Experiment Configuration ---")
        print(json.dumps(new_config, indent=2))
        
        # 4. Execute the orchestrator
        print("Executing python orchestrator.py...")
        subprocess.run(["python", "orchestrator.py"])
        
        # 5. Pause for 5 seconds before the next iteration
        print("Trial concluded. Waiting 5 seconds before the next experiment...")
        time.sleep(5)

if __name__ == '__main__':
    main()
