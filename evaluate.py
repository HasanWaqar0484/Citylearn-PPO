import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from citylearn.citylearn import CityLearnEnv
from env_wrapper import CustomCityLearnWrapper
from ppo_agent import ActorCritic

def evaluate():
    dataset_name = 'citylearn_challenge_2022_phase_1'
    env = CityLearnEnv(dataset_name)
    env = CustomCityLearnWrapper(env)
    
    # Dimensions (must match training)
    num_buildings = len(env.observation_space)
    state_dim = sum([space.shape[0] for space in env.observation_space])
    action_dim = sum([space.shape[0] for space in env.action_space])
    
    # Find the latest model file
    model_files = glob.glob('ppo_citylearn_*.pth')
    if not model_files:
        print("No model files found. Please run training first.")
        return
    
    # Sort by episode number and get the latest
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_model = model_files[-1]
    print(f"Loading model: {latest_model}")
    
    # Load model
    policy = ActorCritic(state_dim, action_dim)
    try:
        policy.load_state_dict(torch.load(latest_model))
        print("Model loaded successfully.")
    except:
        print("Could not load model. Ensure training has run.")
        return
    
    policy.eval()
    
    obs_tuple = env.reset()
    # Handle tuple return from reset
    if isinstance(obs_tuple, tuple):
        state = obs_tuple[0]
    else:
        state = obs_tuple
    
    # Flatten state (concatenate all building observations)
    state = np.concatenate([np.array(s) for s in state])
    state = torch.FloatTensor(state).unsqueeze(0)
    
    rewards = []
    
    done = False
    step_count = 0
    while not done:
        action, _ = policy.get_action(state)
        
        # Split action back into per-building actions
        action_np = action.detach().numpy().flatten()
        action_list = []
        idx = 0
        for space in env.action_space:
            action_size = space.shape[0]
            action_list.append(action_np[idx:idx+action_size])
            idx += action_size
        
        # Step environment
        step_result = env.step(action_list)
        
        # Handle different return formats
        if len(step_result) == 5:
            next_state, reward, done, truncated, info = step_result
            done = done or truncated
        else:
            next_state, reward, done, info = step_result
             
        # Flatten next_state
        next_state = np.concatenate([np.array(s) for s in next_state])
        scalar_reward = sum(reward) if isinstance(reward, list) else reward
        
        rewards.append(scalar_reward)
        state = torch.FloatTensor(next_state).unsqueeze(0)
        step_count += 1
        
    print(f"Evaluation completed!")
    print(f"Total Steps: {step_count}")
    print(f"Total Reward: {sum(rewards):.2f}")
    print(f"Average Reward per Step: {np.mean(rewards):.2f}")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.title("Evaluation Rewards Over Time")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig("evaluation_plot.png")
    print("Plot saved to evaluation_plot.png")
    plt.close()

if __name__ == '__main__':
    evaluate()
