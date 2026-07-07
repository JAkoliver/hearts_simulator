import os
import json
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
    
    # Select which parameter to mutate
    choices = ['learning_rate', 'update_timestep', 'gamma', 'eps_clip']
    choice = random.choice(choices)
    
    if choice == 'learning_rate':
        multiplier = random.choice([0.8, 1.2])
        new_config['learning_rate'] = new_config['learning_rate'] * multiplier
        
    elif choice == 'update_timestep':
        # Adjust games_per_worker by +/- 5, and recalculate update_timestep
        delta = random.choice([-5, 5])
        new_config['games_per_worker'] = max(5, new_config['games_per_worker'] + delta) # prevent zero or negative
        new_config['update_timestep'] = new_config['games_per_worker'] * new_config['max_workers']
        
    elif choice == 'gamma':
        delta = random.choice([-0.01, 0.01])
        new_config['gamma'] = min(1.0, max(0.0, new_config['gamma'] + delta))
        
    elif choice == 'eps_clip':
        delta = random.choice([-0.01, 0.01])
        new_config['eps_clip'] = max(0.01, new_config['eps_clip'] + delta)
        
    return new_config

def is_config_failed(new_config, failed_configs):
    # Check if this exact config dictionary exists in failed configs
    # JSON serialization sorts keys to ensure exact matching, or we can just compare dicts
    for failed_cfg in failed_configs:
        if failed_cfg == new_config:
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
