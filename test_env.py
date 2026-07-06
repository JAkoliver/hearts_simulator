import hearts_env
import random

# 1. Instantiate the C++ Environment
env = hearts_env.HeartsEnv(seed=42)

# 2. Reset the board
env.reset()

print("Environment Initialized.")

# 3. Play a random game to completion
done = False
while not done:
    # Get the 156-float tensor from C++
    observation = env.observe() 
    
    # Get legal actions from C++
    legal_actions_raw = env.get_legal_actions()
    
    # Filter out the -1 sentinel values
    legal_actions = [a for a in legal_actions_raw if a != -1] 
    
    # Choose a random legal move (Later, PyTorch will make this choice!)
    action = random.choice(legal_actions)
    
    # Execute the move in C++
    result = env.step(action)
    done = result.done

print(f"Game Complete! Final action reward: {result.reward}")
