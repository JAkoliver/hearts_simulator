import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import hearts_env
from hearts_net import HeartsNet
import tqdm

EVAL_GAMES = 10000

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
        x = F.relu(self.shared_fc1(observation))
        x = F.relu(self.shared_fc2(x))
        state_value = self.value_head(x)
        logits = self.policy_head(x)
        masked_logits = logits.masked_fill(~legal_actions_mask, float('-inf'))
        return masked_logits, state_value

def load_model(path):
    """Load a checkpoint, auto-detecting v1 vs v2 architecture from its keys."""
    sd = torch.load(path, weights_only=True)
    if 'shared_fc1.weight' in sd:
        net = LegacyHeartsNet()
        arch = "v1"
    else:
        net = HeartsNet()
        arch = "v2"
    net.load_state_dict(sd)
    net.eval()
    return net, arch

def get_card_details(action_id):
    suit = action_id // 13
    rank = (action_id % 13) + 2
    return suit, rank

def get_hand(observation):
    hand = []
    for i in range(52):
        if observation[i] == 1.0:
            hand.append(i)
    return hand

def get_winner_rank(current_trick_cards, led_suit):
    winner_rank = -1
    for c in current_trick_cards:
        s, r = get_card_details(c)
        if s == led_suit and r > winner_rank:
            winner_rank = r
    return winner_rank

def select_action(network, observation, legal_actions_raw):
    state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.zeros((1, 52), dtype=torch.bool)
    for a in legal_actions_raw:
        if a != -1:
            mask_tensor[0, a] = True
            
    with torch.no_grad():
        masked_logits, _ = network(state_tensor, mask_tensor)
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        
    return action.item()

def main():
    print("Initializing Analyzer...")
    
    if not os.path.exists("eval_models"):
        os.makedirs("eval_models")
        print("Created 'eval_models' directory. Please place .pth files inside and run again.")
        return

    model_files = glob.glob(os.path.join("eval_models", "*.pth"))
    if not model_files:
        print("No .pth files found in 'eval_models'. Please add some and run again.")
        return

    num_models = len(model_files)
    if num_models > 4:
        raise ValueError(f"Found {num_models} models, but standard Hearts only supports up to 4.")

    print(f"Found {num_models} models:")
    for m in model_files:
        print(f" - {os.path.basename(m)}")

    # Deterministic Seating Logic
    seats = []
    if num_models == 1:
        seats = [model_files[0]] * 4
    elif num_models == 2:
        model_files.sort(key=os.path.getmtime)
        older_model = model_files[0]
        newer_model = model_files[1]
        print(f"\nAuto-detected 2 models. Setting up 1v3 matchup:")
        print(f" Older (3 seats): {os.path.basename(older_model)}")
        print(f" Newer (1 seat):  {os.path.basename(newer_model)}")
        seats = [newer_model, older_model, older_model, older_model]
    elif num_models == 3:
        random.shuffle(model_files)
        seats = [model_files[0], model_files[1], model_files[2], model_files[0]]
    elif num_models == 4:
        seats = model_files.copy()

    # Load networks (cache by path so shared seats reuse one instance)
    loaded = {}
    for f in set(seats):
        loaded[f] = load_model(f)
    networks = [loaded[s][0] for s in seats]

    print("\nSeating Assignment:")
    for i, s in enumerate(seats):
        print(f" Seat {i}: {os.path.basename(s)} [{loaded[s][1]}]")

    # Initialize Telemetry
    telemetry = {os.path.basename(f): {
        "games_played": 0,
        "total_penalty": 0,
        "qs_dump_opps": 0, "qs_dump_success": 0,
        "duck_opps": 0, "duck_success": 0,
        "spade_bleed_opps": 0, "spade_bleed_success": 0,
        "tricks_won": 0, "tricks_played": 0,
        "forced_win_opps": 0, "forced_win_success": 0
    } for f in model_files}

    env = hearts_env.HeartsEnv(42)

    print(f"\nRunning {EVAL_GAMES} evaluation games...")
    
    for game in tqdm.tqdm(range(EVAL_GAMES), desc="Simulating Games", unit="game", ncols=80, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        env.reset()
        done = False
        
        current_trick_cards = []
        
        while not done:
            player_idx = env.get_current_player()
            model_name = os.path.basename(seats[player_idx])
            obs = env.observe()
            legal_raw = env.get_legal_actions()
            
            hand = get_hand(obs)
            legal_actions = [a for a in legal_raw if a != -1]
            
            # --- EVALUATE TELEMETRY CONDITIONS ---
            
            # Metric: Take Control Rate
            if len(current_trick_cards) == 0:
                for s in seats:
                    telemetry[os.path.basename(s)]["tricks_played"] += 1
            
            led_suit = current_trick_cards[0] // 13 if len(current_trick_cards) > 0 else -1
            winner_rank = get_winner_rank(current_trick_cards, led_suit) if led_suit != -1 else -1

            # Determine action
            action = select_action(networks[player_idx], obs, legal_raw)
            a_suit, a_rank = get_card_details(action)

            # Metric: QS Dumping Efficiency (QS is action 36)
            # Require the QS to actually be playable (excludes e.g. the
            # first trick, where the engine forbids penalty cards)
            if 36 in legal_actions:
                if len(current_trick_cards) > 0:
                    is_void_in_led = all((c // 13) != led_suit for c in hand)
                    qs_can_dump = False
                    
                    if is_void_in_led:
                        qs_can_dump = True
                    elif led_suit == 2:
                        # Spade trick, check if higher spade (AS=38, KS=37) played
                        if 37 in current_trick_cards or 38 in current_trick_cards:
                            qs_can_dump = True
                            
                    if qs_can_dump:
                        telemetry[model_name]["qs_dump_opps"] += 1
                        if action == 36:
                            telemetry[model_name]["qs_dump_success"] += 1

            # Metric: Optimal Ducking Rate
            if len(current_trick_cards) > 0 and a_suit == led_suit and a_rank < winner_rank:
                safe_cards = [c for c in legal_actions if (c // 13) == led_suit and ((c % 13) + 2) < winner_rank]
                if len(safe_cards) > 0:
                    telemetry[model_name]["duck_opps"] += 1
                    max_safe_rank = max((c % 13) + 2 for c in safe_cards)
                    if a_rank == max_safe_rank:
                        telemetry[model_name]["duck_success"] += 1

            # Metric: Spade Bleeding Execution
            if len(current_trick_cards) == 0: # model is leading the trick
                if 36 not in hand:
                    low_spades = [c for c in hand if (c // 13) == 2 and c < 36]
                    # A low spade must actually be leadable (excludes the
                    # forced 2-of-clubs lead on the first trick)
                    if len(low_spades) >= 2 and any(c in legal_actions for c in low_spades):
                        telemetry[model_name]["spade_bleed_opps"] += 1
                        if a_suit == 2 and action < 36:
                            telemetry[model_name]["spade_bleed_success"] += 1

            # Metric: Forced Win Optimization — the model is forced to head the
            # trick (must follow suit and every legal card beats the current
            # winner); optimal play is to shed the highest card while stuck.
            # NOTE: leading a trick is NOT a forced win and is not counted.
            forced = False
            if len(current_trick_cards) > 0 and all((c // 13) == led_suit for c in legal_actions):
                min_legal_rank = min((c % 13) + 2 for c in legal_actions)
                if min_legal_rank > winner_rank:
                    forced = True
                    
            if forced:
                telemetry[model_name]["forced_win_opps"] += 1
                max_legal_rank = max((c % 13) + 2 for c in legal_actions)
                if a_rank == max_legal_rank:
                    telemetry[model_name]["forced_win_success"] += 1

            # --- EXECUTE ACTION ---
            res = env.step(action)
            current_trick_cards.append(action)
            done = res.done
            
            if len(current_trick_cards) == 4:
                # trick ended, determine winner
                t_winner_offset = 0
                highest_r = -1
                for offset, c in enumerate(current_trick_cards):
                    s, r = get_card_details(c)
                    if s == led_suit and r > highest_r:
                        highest_r = r
                        t_winner_offset = offset
                
                trick_leader = (player_idx + 1) % 4
                actual_winner_idx = (trick_leader + t_winner_offset) % 4
                winner_model = os.path.basename(seats[actual_winner_idx])
                
                telemetry[winner_model]["tricks_won"] += 1
                current_trick_cards = []

        # End of Game
        scores = env.get_round_scores()
        for i, s in enumerate(seats):
            m_name = os.path.basename(s)
            telemetry[m_name]["total_penalty"] += scores[i]
            telemetry[m_name]["games_played"] += 1

    # --- AGGREGATE AND PRINT RESULTS ---
    print("\n" + "="*70)
    print(" " * 20 + "TACTICAL TELEMETRY REPORT")
    print("="*70 + "\n")
    
    w_mod = 34
    w_col = 11
    
    def truncate_name(name, width):
        if len(name) > width:
            return name[:width-10] + "..." + name[-7:]
        return name

    # --- TABLE 1 ---
    header1 = f"{'Model':<{w_mod}} | {'Avg Pts':>{w_col}} | {'QS Dump':>{w_col}} | {'Duck Rate':>{w_col}}"
    print(header1)
    print("-" * len(header1))
    
    for model, t in telemetry.items():
        if t["games_played"] == 0: continue
        avg_penalty = t["total_penalty"] / t["games_played"]
        qs_dump = (t["qs_dump_success"] / t["qs_dump_opps"] * 100) if t["qs_dump_opps"] > 0 else 0.0
        duck = (t["duck_success"] / t["duck_opps"] * 100) if t["duck_opps"] > 0 else 0.0
        mod_name = truncate_name(model, w_mod)
        print(f"{mod_name:<{w_mod}} | {avg_penalty:>{w_col}.2f} | {qs_dump:>{w_col-1}.1f}% | {duck:>{w_col-1}.1f}%")

    print("\n")

    # --- TABLE 2 ---
    header2 = f"{'Model':<{w_mod}} | {'Spd Bleed':>{w_col}} | {'Control':>{w_col}} | {'Frc Win':>{w_col}}"
    print(header2)
    print("-" * len(header2))
    
    for model, t in telemetry.items():
        if t["games_played"] == 0: continue
        spade_bleed = (t["spade_bleed_success"] / t["spade_bleed_opps"] * 100) if t["spade_bleed_opps"] > 0 else 0.0
        take_control = (t["tricks_won"] / t["tricks_played"] * 100) if t["tricks_played"] > 0 else 0.0
        forced_win = (t["forced_win_success"] / t["forced_win_opps"] * 100) if t["forced_win_opps"] > 0 else 0.0
        mod_name = truncate_name(model, w_mod)
        print(f"{mod_name:<{w_mod}} | {spade_bleed:>{w_col-1}.1f}% | {take_control:>{w_col-1}.1f}% | {forced_win:>{w_col-1}.1f}%")
        
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
