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
import hearts_env
from hearts_net import HeartsNet, net_from_checkpoint

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
# 2. Batched Action Selection
# ---------------------------------------------------------
def select_actions_batch(network, obs_np, masks_np, device):
    """Sample actions for a batch of observations in one forward pass.

    Rollouts never backprop, so autograd is skipped entirely and the forward
    runs in bf16 (sampling from logits is insensitive to that precision; the
    log_probs recorded here are the SAME ones the PPO ratio uses, so there is
    no old/new precision mismatch). Returns numpy (actions int64,
    log_probs f32, values f32)."""
    obs = torch.from_numpy(obs_np).to(device)
    mask = torch.from_numpy(masks_np).to(device)
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                masked_logits, state_values = network(obs, mask)
            masked_logits = masked_logits.float()
            state_values = state_values.float()
        else:
            masked_logits, state_values = network(obs, mask)
        # -inf logits on illegal actions give them exactly 0 probability
        dist = Categorical(logits=masked_logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
    return (actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            state_values.squeeze(-1).cpu().numpy())

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

def ppo_update(network, optimizer, buffer, device, gamma=1.0, eps_clip=0.2, k_epochs=4,
               minibatch_size=2048, gae_lambda=0.95, entropy_coef=0.01, aux_coef=0.5,
               actor_coef=1.0):
    if len(buffer.states) == 0:
        return None, None

    rewards_np = np.asarray(buffer.rewards, dtype=np.float32)
    values_np = np.asarray(buffer.values, dtype=np.float32)
    dones_np = np.asarray(buffer.dones, dtype=np.bool_)

    advantages_np, returns_np = compute_gae(rewards_np, values_np, dones_np, gamma, gae_lambda)

    # Critic health metric: 1.0 = value head perfectly predicts targets,
    # <= 0.0 = no better than predicting the mean
    explained_var = 1.0 - np.var(returns_np - values_np) / (np.var(returns_np) + 1e-8)

    # Go through numpy: torch.tensor() on large lists is very slow
    old_states = torch.from_numpy(np.asarray(buffer.states, dtype=np.float32)).to(device)
    old_actions = torch.from_numpy(np.asarray(buffer.actions, dtype=np.int64)).to(device)
    old_log_probs = torch.from_numpy(np.asarray(buffer.log_probs, dtype=np.float32)).to(device)
    old_masks = torch.from_numpy(np.asarray(buffer.masks, dtype=np.bool_)).to(device)
    returns = torch.from_numpy(returns_np).to(device)
    advantages = torch.from_numpy(advantages_np).to(device)
    hand_labels = torch.from_numpy(np.asarray(buffer.hand_labels, dtype=np.float32)).to(device)

    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = old_states.shape[0]
    belief_bce_sum, belief_bce_count = 0.0, 0
    for epoch in range(k_epochs):
        perm = torch.randperm(n, device=device)
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

            # actor_coef 0 = critic warmup: the value/belief heads adapt to
            # the on-policy distribution while the policy stays untouched, so
            # early garbage advantages can't damage a good warm-start policy
            loss = (actor_coef * (actor_loss - entropy_coef * entropy.mean())
                    + 0.5 * critic_loss + aux_coef * belief_loss)

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
# 4. Vectorized Rollouts
# ---------------------------------------------------------
# One process, num_envs environments stepped in lockstep. Each lockstep round
# groups the envs by which network decides next and runs ONE batched forward
# per network on the device. This replaces the old ProcessPoolExecutor with
# batch-1 CPU inference per decision: no state-dict shipping to workers, and
# the GPU sees real batches. Buffer/GAE/terminal-reward semantics are
# unchanged from the multiprocessing version.

class EnvSlot:
    __slots__ = ('env', 'nets', 'buffers', 'steps_this_game')

    def __init__(self, seed):
        self.env = hearts_env.HeartsEnv(seed=seed)
        self.env.reset()  # a freshly constructed env has no deal yet
        self.nets = None
        self.buffers = [RolloutBuffer() for _ in range(4)]
        self.steps_this_game = 0

def assign_seats(slot, main_network, active_pool):
    """Seat 0 is always the learner; each other seat is 50% a pool opponent."""
    nets = [main_network]
    for _ in range(1, 4):
        if active_pool and random.random() < 0.5:
            nets.append(random.choice(active_pool))
        else:
            nets.append(main_network)
    slot.nets = nets

def run_cycle(slots, main_network, active_pool, games_target, device):
    """Play games across all slots until games_target games complete.

    Returns (games_played, p0_reward_sum, p0_raw_score_sum). Decision records
    for the learner's seats accumulate in each slot's buffers."""
    games_done = 0
    p0_reward_sum = 0.0
    p0_raw_sum = 0.0

    while True:
        # Once the target is reached, DRAIN: only step envs that are mid-game
        # so every recorded trajectory ends with its true terminal reward.
        # Merging a truncated episode would train its tail on garbage credit
        # (GAE bootstraps 0 past the buffer end).
        draining = games_done >= games_target

        # Group envs by the network that decides their next action
        groups = {}
        for k, slot in enumerate(slots):
            if draining and slot.steps_this_game == 0:
                continue
            net = slot.nets[slot.env.get_current_player()]
            key = id(net)
            if key not in groups:
                groups[key] = (net, [])
            groups[key][1].append(k)
        if not groups:
            break

        for net, idxs in groups.values():
            n = len(idxs)
            obs_np = np.empty((n, 550), dtype=np.float32)
            masks_np = np.zeros((n, 52), dtype=np.bool_)
            for j, k in enumerate(idxs):
                obs_np[j] = slots[k].env.observe()
                for a in slots[k].env.get_legal_actions():
                    if a != -1:
                        masks_np[j, a] = True

            actions, log_probs, values = select_actions_batch(net, obs_np, masks_np, device)

            for j, k in enumerate(idxs):
                slot = slots[k]
                pid = slot.env.get_current_player()
                is_main = slot.nets[pid] is main_network

                # Belief labels must be captured BEFORE stepping: they are
                # relative to the acting player and describe the state being
                # recorded
                labels = slot.env.observe_opponent_hands() if is_main else None

                done = slot.env.step(int(actions[j])).done
                slot.steps_this_game += 1

                if is_main:
                    b = slot.buffers[pid]
                    b.states.append(obs_np[j])
                    b.actions.append(int(actions[j]))
                    b.log_probs.append(float(log_probs[j]))
                    b.rewards.append(0.0)
                    b.values.append(float(values[j]))
                    b.masks.append(masks_np[j])
                    b.dones.append(False)
                    b.hand_labels.append(labels)

                if done:
                    scores = slot.env.get_round_scores()
                    avg = sum(scores) / 4.0
                    rel_rewards = [avg - s for s in scores]
                    p0_reward_sum += rel_rewards[0]
                    p0_raw_sum += scores[0]
                    # Only assign the terminal reward to seats the learner
                    # actually played this game; otherwise rewards[-1]
                    # belongs to an earlier game
                    for i in range(4):
                        if slot.nets[i] is main_network and len(slot.buffers[i].rewards) > 0:
                            slot.buffers[i].rewards[-1] = rel_rewards[i]
                            slot.buffers[i].dones[-1] = True
                    games_done += 1
                    slot.env.reset()
                    assign_seats(slot, main_network, active_pool)
                    slot.steps_this_game = 0

    return games_done, p0_reward_sum, p0_raw_sum

# ---------------------------------------------------------
# 4b. Vectorized Rollouts (HeartsVecEnv)
# ---------------------------------------------------------
# Same semantics as run_cycle, but all per-env work (observations, masks,
# labels, stepping) happens in C++ through batched numpy arrays - the
# measured bottleneck was pybind list marshaling at ~16us/decision.
# seat_net maps (env, seat) -> index into a STABLE net registry (0 = the
# learner); games straddle cycle boundaries, so assignments must not
# reference per-cycle pool subsets.

def run_cycle_vec(vec, registry, active_ids, seat_net, steps_this_game,
                  buffers, games_target, device):
    n = vec.size()
    env_ids = np.arange(n, dtype=np.int64)
    games_done = 0
    p0_reward_sum = 0.0
    p0_raw_sum = 0.0

    while True:
        draining = games_done >= games_target
        if draining:
            active = env_ids[steps_this_game > 0]
            if active.size == 0:
                break
        else:
            active = env_ids

        cp = vec.current_players()
        deciding = seat_net[env_ids, cp]  # registry index per env

        for k in np.unique(deciding[active]):
            g = active[deciding[active] == k]
            obs = vec.observe_batch(g)
            mask = vec.legal_mask_batch(g)
            actions, log_probs, values = select_actions_batch(
                registry[k], obs, mask, device)
            is_learner = (k == 0)
            if is_learner:
                # Belief labels must be captured BEFORE stepping
                labels = vec.labels_batch(g)

            dones, scores = vec.step_batch(g, actions.astype(np.int64))
            steps_this_game[g] += 1

            if is_learner:
                for j in range(len(g)):
                    b = buffers[int(g[j])][int(cp[g[j]])]
                    b.states.append(obs[j])
                    b.actions.append(int(actions[j]))
                    b.log_probs.append(float(log_probs[j]))
                    b.rewards.append(0.0)
                    b.values.append(float(values[j]))
                    b.masks.append(mask[j])
                    b.dones.append(False)
                    b.hand_labels.append(labels[j])

            for j in np.flatnonzero(dones):
                e = int(g[j])
                sc = scores[j]
                avg = float(sc.sum()) / 4.0
                p0_reward_sum += avg - float(sc[0])
                p0_raw_sum += float(sc[0])
                # Only learner seats that actually played this game carry the
                # terminal reward (same rule as the slot path)
                for seat in range(4):
                    if seat_net[e, seat] == 0 and len(buffers[e][seat].rewards) > 0:
                        buffers[e][seat].rewards[-1] = avg - float(sc[seat])
                        buffers[e][seat].dones[-1] = True
                games_done += 1
                steps_this_game[e] = 0
                for seat in range(1, 4):
                    seat_net[e, seat] = (random.choice(active_ids)
                                         if active_ids and random.random() < 0.5 else 0)

    return games_done, p0_reward_sum, p0_raw_sum

# ---------------------------------------------------------
# 5. Main Training Loop
# ---------------------------------------------------------
def main():
    print("Initializing Hearts PPO Training Pipeline (vectorized)...")

    with open('config.json', 'r') as f:
        config = json.load(f)

    device = torch.device(config.get('device',
                                     'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")
    torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    # TF32 matmuls: ~1.5x on Ada for the fp32 PPO update path, precision is
    # ample for RL losses
    torch.set_float32_matmul_precision('high')

    # train_init lets a fine-tune start from a different checkpoint than the
    # gate baseline (e.g. the big distilled net). Preference: once a
    # train_init-shaped net has been PROMOTED (hearts_model_final.pth has the
    # same architecture), continue from the promoted weights; until then,
    # bootstrap from train_init. Without train_init: normal resume order.
    def ckpt_shape(path):
        sd = torch.load(path, weights_only=True, map_location='cpu')
        key = 'input_fc.weight' if 'input_fc.weight' in sd else 'card_embed.weight'
        depth = sum(k.startswith(('blocks.', 'enc_blocks.')) for k in sd)
        return key, tuple(sd[key].shape), depth

    init_path = None
    ti = config.get('train_init')
    if ti and os.path.exists(ti):
        if (os.path.exists('hearts_model_final.pth')
                and ckpt_shape('hearts_model_final.pth') == ckpt_shape(ti)):
            init_path = 'hearts_model_final.pth'
        else:
            init_path = ti
    else:
        for candidate in ['hearts_model_final.pth', 'hearts_model_interrupted.pth']:
            if os.path.exists(candidate):
                init_path = candidate
                break

    if init_path:
        network = net_from_checkpoint(init_path)
        n_params = sum(p.numel() for p in network.parameters())
        print(f"Model Resumption Successful: Loaded {init_path} "
              f"({type(network).__name__}, {n_params / 1e6:.2f}M params)")
    else:
        network = HeartsNet()
        print("Fresh default-size network")
    network.to(device)

    optimizer = optim.Adam(network.parameters(), lr=config.get('learning_rate', 5e-5))

    # Restore Adam moment estimates from the previous trial so every experiment
    # doesn't start with a cold optimizer on a warm network. A shape mismatch
    # (e.g. after switching network size) falls through to fresh moments.
    if os.path.exists('hearts_optimizer.pth'):
        try:
            optimizer.load_state_dict(torch.load('hearts_optimizer.pth', weights_only=True))
            for group in optimizer.param_groups:
                group['lr'] = config.get('learning_rate', 5e-5)
            print("Optimizer state restored: Adam moments carried over from previous trial.")
        except Exception as e:
            print(f"Could not load optimizer state ({e}); starting with fresh Adam moments.")

    update_timestep = config.get('update_timestep', 560)
    num_envs = config.get('num_envs', 128)
    active_pool_size = config.get('active_pool_size', 4)
    # Critic warmup: freeze the actor for the first N games so the value and
    # belief heads adapt to the on-policy distribution first (a warm-started
    # policy with a cold critic otherwise trains on noise advantages)
    warmup_games = config.get('critic_warmup_games', 20000)
    train_log = open('train_last_run.log', 'w')

    if os.environ.get('SMOKE_TEST') == '1':
        max_episodes = 100
        print("SMOKE_TEST mode active: max_episodes set to 100")
    else:
        max_episodes = config.get('max_episodes', 250000)

    historical_pool = [copy.deepcopy(network)]
    for m_file in glob.glob('Hall_of_Fame/hearts_model_milestone_*.pth'):
        try:
            m_net = net_from_checkpoint(m_file)
        except Exception as e:
            print(f"Skipping unloadable milestone {m_file}: {e}")
            continue
        m_net.to(device)
        m_net.eval()
        historical_pool.append(m_net)
        print(f"Loaded milestone opponent: {m_file}")

    use_vec = bool(config.get('vec_env', True))
    if use_vec:
        # Stable net registry: index 0 is the learner; pool nets get
        # append-only indices so mid-game seat assignments survive both the
        # per-cycle active-subset change and pool trimming.
        registry = [network] + list(historical_pool)
        pool_ids = list(range(1, len(registry)))
        vec = hearts_env.HeartsVecEnv(num_envs, 1000)
        seat_net = np.zeros((num_envs, 4), dtype=np.int64)
        steps_this_game = np.zeros(num_envs, dtype=np.int64)
        vec_buffers = [[RolloutBuffer() for _ in range(4)] for _ in range(num_envs)]
        init_ids = random.sample(pool_ids, min(active_pool_size, len(pool_ids)))
        for e in range(num_envs):
            for seat in range(1, 4):
                seat_net[e, seat] = (random.choice(init_ids)
                                     if init_ids and random.random() < 0.5 else 0)
        print(f"Vectorized rollouts: HeartsVecEnv({num_envs})")
    else:
        slots = [EnvSlot(seed=1000 + i) for i in range(num_envs)]
        print("Slot rollouts (vec_env disabled)")

    games_played = 0
    pool_refresh_interval = config.get('pool_refresh_interval', 25000)
    next_pool_refresh = pool_refresh_interval

    try:
        while games_played < max_episodes:
            # A small random subset of the pool plays this cycle so per-step
            # inference batches stay large (fewer distinct nets per round)
            if use_vec:
                active_ids = random.sample(pool_ids,
                                           min(active_pool_size, len(pool_ids)))
                done_games, p0_reward_sum, p0_raw_sum = run_cycle_vec(
                    vec, registry, active_ids, seat_net, steps_this_game,
                    vec_buffers, update_timestep, device)
            else:
                active_pool = random.sample(historical_pool,
                                            min(active_pool_size, len(historical_pool)))
                for slot in slots:
                    if slot.nets is None:
                        assign_seats(slot, network, active_pool)
                done_games, p0_reward_sum, p0_raw_sum = run_cycle(
                    slots, network, active_pool, update_timestep, device)

            in_warmup = games_played < warmup_games
            games_played += done_games
            if in_warmup and games_played >= warmup_games:
                print(f"Critic warmup complete at {games_played} games; actor unfrozen.")
            avg_p0_reward = p0_reward_sum / done_games
            avg_p0_raw = p0_raw_sum / done_games

            # Merge all seats into one dataset: updating per-seat sequentially
            # runs later seats' ratios against log_probs that are already stale
            merged_buffer = RolloutBuffer()
            if use_vec:
                for env_bufs in vec_buffers:
                    for b in env_bufs:
                        merged_buffer.extend(b)
                        b.clear()
            else:
                for slot in slots:
                    for i in range(4):
                        merged_buffer.extend(slot.buffers[i])
                        slot.buffers[i].clear()

            explained_var, belief_bce = ppo_update(network, optimizer, merged_buffer, device,
                                                   gamma=config.get('gamma', 1.0),
                                                   eps_clip=config.get('eps_clip', 0.2),
                                                   k_epochs=config.get('k_epochs', 4),
                                                   minibatch_size=config.get('minibatch_size', 2048),
                                                   gae_lambda=config.get('gae_lambda', 0.95),
                                                   entropy_coef=config.get('entropy_coef', 0.01),
                                                   aux_coef=config.get('aux_coef', 0.5),
                                                   actor_coef=0.0 if in_warmup else 1.0)

            ev_str = f"{explained_var:.3f}" if explained_var is not None else "n/a"
            bce_str = f"{belief_bce:.4f}" if belief_bce is not None else "n/a"
            line = (f"Games: {games_played} | Pool: {len(historical_pool)} | "
                    f"P0 Raw: {avg_p0_raw:.2f} | P0 Rel: {avg_p0_reward:+.2f} | "
                    f"Critic EV: {ev_str} | Belief BCE: {bce_str}"
                    + (" | WARMUP" if in_warmup else ""))
            print(line)
            train_log.write(line + "\n")
            train_log.flush()

            # Periodically snapshot the current policy into the opponent pool so
            # in-trial opponents track progress instead of staying frozen
            if games_played >= next_pool_refresh:
                snap = copy.deepcopy(network)
                snap.eval()
                historical_pool.append(snap)
                if use_vec:
                    registry.append(snap)
                    pool_ids.append(len(registry) - 1)
                if len(historical_pool) > 50:
                    historical_pool.pop(0)
                    if use_vec:
                        # registry keeps the entry (in-flight games may still
                        # reference it); it just stops being sampled
                        pool_ids.pop(0)
                next_pool_refresh += pool_refresh_interval

        print("\nTraining Complete! Saving final model...")
        torch.save(network.cpu().state_dict(), 'hearts_model_final.pth')
        torch.save(optimizer.state_dict(), 'hearts_optimizer.pth')
        print("Model saved to hearts_model_final.pth!")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Model saved safely!")
        torch.save(network.cpu().state_dict(), 'hearts_model_interrupted.pth')
        torch.save(optimizer.state_dict(), 'hearts_optimizer.pth')

if __name__ == '__main__':
    main()
