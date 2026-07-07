import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import copy
import random
import os
import glob
import psutil
import multiprocessing
import concurrent.futures
import hearts_env
from hearts_net import HeartsNet

# ---------------------------------------------------------
# 1. The Memory Buffer
# ---------------------------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.masks = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.masks.clear()
        
    def extend(self, other):
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.log_probs.extend(other.log_probs)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.masks.extend(other.masks)

# ---------------------------------------------------------
# 2. Action Selection Function
# ---------------------------------------------------------
def select_action(network, observation, legal_actions_raw):
    # Convert 181-float observation list to tensor
    state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0) # Shape: (1, 181)
    
    # Create boolean mask for legal actions (True = legal, False = illegal)
    mask_tensor = torch.zeros((1, 52), dtype=torch.bool)
    for a in legal_actions_raw:
        if a != -1:
            mask_tensor[0, a] = True
            
    # Forward pass
    masked_logits, state_value = network(state_tensor, mask_tensor)
    
    # Because masked_logits has -inf for illegal actions, 
    # Categorical will beautifully assign them 0 probability.
    dist = Categorical(logits=masked_logits)
    action = dist.sample()
    
    return action.item(), dist.log_prob(action).detach(), state_value.detach(), mask_tensor

# ---------------------------------------------------------
# 3. PPO Update Function
# ---------------------------------------------------------
def ppo_update(network, optimizer, buffer, gamma=1.0, eps_clip=0.2, k_epochs=4):
    if len(buffer.states) == 0:
        return
        
    returns = []
    discounted_reward = 0
    # Process rewards in reverse
    for reward in reversed(buffer.rewards):
        discounted_reward = reward + (gamma * discounted_reward)
        returns.insert(0, discounted_reward)
        
    old_states = torch.tensor(buffer.states, dtype=torch.float32)
    old_actions = torch.tensor(buffer.actions, dtype=torch.float32)
    old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)
    old_masks = torch.tensor(buffer.masks, dtype=torch.bool)
    returns = torch.tensor(returns, dtype=torch.float32)
    old_values = torch.tensor(buffer.values, dtype=torch.float32)
    
    advantages = returns - old_values.detach()
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    for _ in range(k_epochs):
        masked_logits, state_values = network(old_states, old_masks)
        dist = Categorical(logits=masked_logits)
        
        new_log_probs = dist.log_prob(old_actions)
        entropy = dist.entropy()
        
        ratios = torch.exp(new_log_probs - old_log_probs.detach())
        
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
        
        actor_loss = -torch.min(surr1, surr2).mean()
        
        critic_loss = nn.MSELoss()(state_values.squeeze(), returns)
        
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ---------------------------------------------------------
# 4. Rollout Worker (Multiprocessing)
# ---------------------------------------------------------
def rollout_worker(network_state_dict, historical_pool_state_dicts, num_games, env_seed):
    torch.set_num_threads(1)
    # Initialize environment and networks locally
    env = hearts_env.HeartsEnv(seed=env_seed)
    
    main_network = HeartsNet()
    main_network.load_state_dict(network_state_dict)
    
    historical_networks = []
    for sd in historical_pool_state_dicts:
        net = HeartsNet()
        net.load_state_dict(sd)
        historical_networks.append(net)
        
    buffer_list = [RolloutBuffer() for _ in range(4)]
    p0_reward_sum = 0.0
    p0_raw_score_sum = 0.0
    
    for _ in range(num_games):
        env.reset()
        done = False
        
        player_networks = []
        player_networks.append(main_network)
        for i in range(1, 4):
            if historical_networks and random.random() < 0.5:
                player_networks.append(random.choice(historical_networks))
            else:
                player_networks.append(main_network)
                
        while not done:
            obs = env.observe()
            legal_actions = env.get_legal_actions()
            player_id = env.get_current_player()
            
            action, log_prob, state_value, mask = select_action(player_networks[player_id], obs, legal_actions)
            
            result = env.step(action)
            done = result.done
            
            if player_networks[player_id] == main_network:
                buffer_list[player_id].states.append(obs)
                buffer_list[player_id].actions.append(action)
                buffer_list[player_id].log_probs.append(log_prob.item())
                buffer_list[player_id].rewards.append(0.0) 
                buffer_list[player_id].values.append(state_value.item())
                buffer_list[player_id].masks.append(mask.tolist()[0])
                
        # End of round
        scores = env.get_round_scores()
        avg = sum(scores) / 4.0
        rel_rewards = [avg - score for score in scores]
        
        p0_reward_sum += rel_rewards[0]
        p0_raw_score_sum += scores[0]
        
        for i in range(4):
            if len(buffer_list[i].rewards) > 0:
                buffer_list[i].rewards[-1] = rel_rewards[i]
                
    return buffer_list, p0_reward_sum, p0_raw_score_sum

# ---------------------------------------------------------
# 5. Main Training Loop
# ---------------------------------------------------------
def main():
    print("Initializing Hearts PPO Training Pipeline (Multiprocessing)...")
    
    network = HeartsNet()
    if os.path.exists('hearts_model_interrupted.pth'):
        network.load_state_dict(torch.load('hearts_model_interrupted.pth', weights_only=True))
        print("Model Resumption Successful: Loaded weights from hearts_model_interrupted.pth!")
    elif os.path.exists('hearts_model_final.pth'):
        network.load_state_dict(torch.load('hearts_model_final.pth', weights_only=True))
        print("Model Resumption Successful: Loaded weights from hearts_model_final.pth!")
        
    optimizer = optim.Adam(network.parameters(), lr=5e-5)
    
    update_timestep = 560
    games_per_worker = 40
    max_workers = 14
    max_episodes = 15160000
    
    historical_pool = []
    for _ in range(5):
        historical_pool.append(copy.deepcopy(network))
        
    for m_file in glob.glob('hearts_model_milestone_*.pth'):
        m_net = HeartsNet()
        m_net.load_state_dict(torch.load(m_file, weights_only=True))
        historical_pool.append(m_net)
        print(f"Loaded milestone opponent: {m_file}")

    games_played = 0
    baseline_games = 17000000
    next_milestone = 1000000
    
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    try:
        while games_played < max_episodes:
            # Prepare state dicts for workers
            network_sd = network.state_dict()
            historical_sds = [net.state_dict() for net in historical_pool]
            
            futures = []
            for i in range(max_workers):
                seed = games_played + i
                futures.append(executor.submit(
                    rollout_worker, 
                    network_sd, 
                    historical_sds, 
                    games_per_worker, 
                    seed
                ))
                
            master_buffer_list = [RolloutBuffer() for _ in range(4)]
            avg_p0_reward_batch = 0.0
            avg_p0_raw_batch = 0.0
            
            for future in concurrent.futures.as_completed(futures):
                local_buffers, p0_reward_sum, p0_raw_score_sum = future.result()
                avg_p0_reward_batch += p0_reward_sum
                avg_p0_raw_batch += p0_raw_score_sum
                
                for i in range(4):
                    master_buffer_list[i].extend(local_buffers[i])
                    
            games_played += update_timestep
            avg_p0_reward = avg_p0_reward_batch / update_timestep
            avg_p0_raw = avg_p0_raw_batch / update_timestep
            
            print(f"Games: {games_played} | Pool: {len(historical_pool)} | P0 Raw: {avg_p0_raw:.2f} | P0 Rel: {avg_p0_reward:+.2f} | PPO Update...")
            
            for i in range(4):
                if len(master_buffer_list[i].states) > 0:
                    ppo_update(network, optimizer, master_buffer_list[i])
                    
            if games_played >= next_milestone:
                total_lifetime_games = baseline_games + next_milestone
                torch.save(network.state_dict(), f'hearts_model_milestone_{total_lifetime_games}.pth')
                print(f"Global Milestone saved: hearts_model_milestone_{total_lifetime_games}.pth")
                next_milestone += 1000000
                    
            if games_played % 25000 == 0:
                historical_pool.append(copy.deepcopy(network))
                if len(historical_pool) > 50:
                    historical_pool.pop(0)
                    
        print("\nTraining Complete! Saving final model...")
        torch.save(network.state_dict(), 'hearts_model_final.pth')
        print("Model saved to hearts_model_final.pth!")
        
        print("Automating C++ deployment... Tracing network architecture...")
        network.eval()
        dummy_obs = torch.zeros(1, 181, dtype=torch.float32)
        dummy_mask = torch.zeros(1, 52, dtype=torch.bool)
        traced_script_module = torch.jit.trace(network, (dummy_obs, dummy_mask))
        traced_script_module.save("hearts_ai_grandmaster.pt")
        print("Deployment asset 'hearts_ai_grandmaster.pt' successfully updated!")
        
        executor.shutdown(wait=True)
                        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Model saved safely!")
        torch.save(network.state_dict(), 'hearts_model_interrupted.pth')
        
        print("Shutting down worker pool...")
        executor.shutdown(wait=False, cancel_futures=True)
        
        print("Hard-killing stubborn Windows child processes...")
        parent = psutil.Process(os.getpid())
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

if __name__ == '__main__':
    # Required for Windows multiprocessing compatibility
    multiprocessing.freeze_support() if hasattr(multiprocessing, 'freeze_support') else None
    main()
