import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import copy
import random
import os
import glob
import json
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
        self.dones = []
        self.hand_labels = []  # ground-truth opponent hands for the belief head

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.masks.clear()
        self.dones.clear()
        self.hand_labels.clear()

    def extend(self, other):
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.log_probs.extend(other.log_probs)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.masks.extend(other.masks)
        self.dones.extend(other.dones)
        self.hand_labels.extend(other.hand_labels)

# ---------------------------------------------------------
# 2. Action Selection Function
# ---------------------------------------------------------
def select_action(network, observation, legal_actions_raw):
    # Convert 550-float observation list to tensor
    state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0) # Shape: (1, 550)
    
    # Create boolean mask for legal actions (True = legal, False = illegal)
    mask_tensor = torch.zeros((1, 52), dtype=torch.bool)
    for a in legal_actions_raw:
        if a != -1:
            mask_tensor[0, a] = True

    # Rollouts never backprop, so skip autograd graph construction entirely
    with torch.no_grad():
        masked_logits, state_value = network(state_tensor, mask_tensor)

        # Because masked_logits has -inf for illegal actions,
        # Categorical will beautifully assign them 0 probability.
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

    return action.item(), log_prob, state_value, mask_tensor

# ---------------------------------------------------------
# 3. PPO Update Function
# ---------------------------------------------------------
def compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=0.95):
    """Generalized Advantage Estimation over a buffer of complete episodes.

    Resets the accumulator and bootstrap value at every episode boundary
    (done=True marks the LAST step of an episode), so nothing bleeds across
    games. Returns (advantages, value_targets) as float32 numpy arrays.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for t in range(n - 1, -1, -1):
        if dones[t]:
            next_value = 0.0
            gae = 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages, advantages + values

def ppo_update(network, optimizer, buffer, gamma=1.0, eps_clip=0.2, k_epochs=4,
               minibatch_size=2048, gae_lambda=0.95, entropy_coef=0.01, aux_coef=0.5):
    if len(buffer.states) == 0:
        return None, None

    rewards_np = np.asarray(buffer.rewards, dtype=np.float32)
    values_np = np.asarray(buffer.values, dtype=np.float32)
    dones_np = np.asarray(buffer.dones, dtype=np.bool_)

    advantages_np, returns_np = compute_gae(rewards_np, values_np, dones_np, gamma, gae_lambda)

    # Critic health metric: 1.0 = value head perfectly predicts targets,
    # <= 0.0 = no better than predicting the mean
    explained_var = 1.0 - np.var(returns_np - values_np) / (np.var(returns_np) + 1e-8)

    # Go through numpy: torch.tensor() on large lists-of-lists is very slow
    old_states = torch.from_numpy(np.asarray(buffer.states, dtype=np.float32))
    old_actions = torch.from_numpy(np.asarray(buffer.actions, dtype=np.int64))
    old_log_probs = torch.from_numpy(np.asarray(buffer.log_probs, dtype=np.float32))
    old_masks = torch.from_numpy(np.asarray(buffer.masks, dtype=np.bool_))
    returns = torch.from_numpy(returns_np)
    advantages = torch.from_numpy(advantages_np)
    hand_labels = torch.from_numpy(np.asarray(buffer.hand_labels, dtype=np.float32))

    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = old_states.shape[0]
    belief_bce_sum, belief_bce_count = 0.0, 0
    for epoch in range(k_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, minibatch_size):
            idx = perm[start:start + minibatch_size]

            masked_logits, state_values, belief_logits = network.forward_all(old_states[idx], old_masks[idx])
            dist = Categorical(logits=masked_logits)

            new_log_probs = dist.log_prob(old_actions[idx])
            entropy = dist.entropy()

            ratios = torch.exp(new_log_probs - old_log_probs[idx])

            mb_advantages = advantages[idx]
            surr1 = ratios * mb_advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * mb_advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(state_values.squeeze(-1), returns[idx])

            # Auxiliary belief loss: supervised prediction of opponents' hands
            belief_loss = nn.functional.binary_cross_entropy_with_logits(
                belief_logits, hand_labels[idx])

            loss = (actor_loss + 0.5 * critic_loss
                    - entropy_coef * entropy.mean() + aux_coef * belief_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()

            if epoch == k_epochs - 1:
                belief_bce_sum += belief_loss.item()
                belief_bce_count += 1

    belief_bce = belief_bce_sum / belief_bce_count if belief_bce_count else None
    return explained_var, belief_bce

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

            # Belief labels must be captured BEFORE stepping: they are relative
            # to the acting player and describe the state being recorded
            hand_labels = env.observe_opponent_hands() if player_networks[player_id] is main_network else None

            result = env.step(action)
            done = result.done

            if player_networks[player_id] is main_network:
                buffer_list[player_id].states.append(obs)
                buffer_list[player_id].actions.append(action)
                buffer_list[player_id].log_probs.append(log_prob.item())
                buffer_list[player_id].rewards.append(0.0)
                buffer_list[player_id].values.append(state_value.item())
                buffer_list[player_id].masks.append(mask.tolist()[0])
                buffer_list[player_id].dones.append(False)
                buffer_list[player_id].hand_labels.append(hand_labels)
                
        # End of round
        scores = env.get_round_scores()
        avg = sum(scores) / 4.0
        rel_rewards = [avg - score for score in scores]
        
        p0_reward_sum += rel_rewards[0]
        p0_raw_score_sum += scores[0]
        
        # Only assign the terminal reward to seats the main network actually
        # played this game; otherwise rewards[-1] belongs to an earlier game
        for i in range(4):
            if player_networks[i] is main_network and len(buffer_list[i].rewards) > 0:
                buffer_list[i].rewards[-1] = rel_rewards[i]
                buffer_list[i].dones[-1] = True
                
    return buffer_list, p0_reward_sum, p0_raw_score_sum

# ---------------------------------------------------------
# 5. Main Training Loop
# ---------------------------------------------------------
def main():
    print("Initializing Hearts PPO Training Pipeline (Multiprocessing)...")
    
    network = HeartsNet()
    if os.path.exists('hearts_model_final.pth'):
        network.load_state_dict(torch.load('hearts_model_final.pth', weights_only=True))
        print("Model Resumption Successful: Loaded weights from hearts_model_final.pth!")
    elif os.path.exists('hearts_model_interrupted.pth'):
        network.load_state_dict(torch.load('hearts_model_interrupted.pth', weights_only=True))
        print("Model Resumption Successful: Loaded weights from hearts_model_interrupted.pth!")
        
    with open('config.json', 'r') as f:
        config = json.load(f)

    optimizer = optim.Adam(network.parameters(), lr=config.get('learning_rate', 5e-5))

    # Restore Adam moment estimates from the previous trial so every experiment
    # doesn't start with a cold optimizer on a warm network
    if os.path.exists('hearts_optimizer.pth'):
        try:
            optimizer.load_state_dict(torch.load('hearts_optimizer.pth', weights_only=True))
            # load_state_dict restores the OLD learning rate too; re-apply this
            # trial's (possibly mutated) value
            for group in optimizer.param_groups:
                group['lr'] = config.get('learning_rate', 5e-5)
            print("Optimizer state restored: Adam moments carried over from previous trial.")
        except Exception as e:
            print(f"Could not load optimizer state ({e}); starting with fresh Adam moments.")
    
    update_timestep = config.get('update_timestep', 560)
    games_per_worker = config.get('games_per_worker', 40)
    max_workers = config.get('max_workers', 14)
    
    if os.environ.get('SMOKE_TEST') == '1':
        max_episodes = 100
        print("SMOKE_TEST mode active: max_episodes set to 100")
    else:
        max_episodes = config.get('max_episodes', 250000)
    
    historical_pool = [copy.deepcopy(network)]

    for m_file in glob.glob('Hall_of_Fame/hearts_model_milestone_*.pth'):
        m_net = HeartsNet()
        try:
            m_net.load_state_dict(torch.load(m_file, weights_only=True))
        except RuntimeError as e:
            print(f"Skipping incompatible milestone (old architecture?): {m_file}")
            continue
        historical_pool.append(m_net)
        print(f"Loaded milestone opponent: {m_file}")

    games_played = 0
    pool_refresh_interval = config.get('pool_refresh_interval', 25000)
    next_pool_refresh = pool_refresh_interval

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
            
            # Merge all seats into one dataset: updating per-seat sequentially
            # runs later seats' ratios against log_probs that are already stale
            merged_buffer = RolloutBuffer()
            for i in range(4):
                merged_buffer.extend(master_buffer_list[i])

            explained_var, belief_bce = ppo_update(network, optimizer, merged_buffer,
                                                   gamma=config.get('gamma', 1.0),
                                                   eps_clip=config.get('eps_clip', 0.2),
                                                   k_epochs=config.get('k_epochs', 4),
                                                   minibatch_size=config.get('minibatch_size', 2048),
                                                   gae_lambda=config.get('gae_lambda', 0.95),
                                                   entropy_coef=config.get('entropy_coef', 0.01),
                                                   aux_coef=config.get('aux_coef', 0.5))

            ev_str = f"{explained_var:.3f}" if explained_var is not None else "n/a"
            bce_str = f"{belief_bce:.4f}" if belief_bce is not None else "n/a"
            print(f"Games: {games_played} | Pool: {len(historical_pool)} | P0 Raw: {avg_p0_raw:.2f} | P0 Rel: {avg_p0_reward:+.2f} | Critic EV: {ev_str} | Belief BCE: {bce_str}")

            # Periodically snapshot the current policy into the opponent pool so
            # in-trial opponents track progress instead of staying frozen
            if games_played >= next_pool_refresh:
                historical_pool.append(copy.deepcopy(network))
                if len(historical_pool) > 50:
                    historical_pool.pop(0)
                next_pool_refresh += pool_refresh_interval
                    
        print("\nTraining Complete! Saving final model...")
        torch.save(network.state_dict(), 'hearts_model_final.pth')
        torch.save(optimizer.state_dict(), 'hearts_optimizer.pth')
        print("Model saved to hearts_model_final.pth!")
        
        executor.shutdown(wait=True)
                        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Model saved safely!")
        torch.save(network.state_dict(), 'hearts_model_interrupted.pth')
        torch.save(optimizer.state_dict(), 'hearts_optimizer.pth')
        
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
