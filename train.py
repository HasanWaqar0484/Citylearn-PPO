import os
import torch
import numpy as np
from citylearn.citylearn import CityLearnEnv
from env_wrapper import CustomCityLearnWrapper
from ppo_agent import PPOAgent, Memory

def main():
    # Initialize environment
    dataset_name = 'citylearn_challenge_2022_phase_1'
    
    try:
        env = CityLearnEnv(dataset_name)
    except Exception as e:
        print(f"Could not load dataset {dataset_name}: {e}")
        print("Please ensure the dataset is available or specify a correct path.")
        return

    # Wrap environment
    env = CustomCityLearnWrapper(env)
    
    # Inspect environment structure
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    
    # CityLearn has multiple buildings, each with their own observation and action space
    # For centralized control, we concatenate all observations and actions
    num_buildings = len(env.observation_space)
    state_dim = sum([space.shape[0] for space in env.observation_space])
    action_dim = sum([space.shape[0] for space in env.action_space])
        
    print(f"Number of buildings: {num_buildings}")
    print(f"Total State Dim: {state_dim}, Total Action Dim: {action_dim}")
    
    # Hyperparameters
    lr = 3e-4
    gamma = 0.99
    K_epochs = 10
    eps_clip = 0.2
    
    agent = PPOAgent(state_dim, action_dim, lr, gamma, eps_clip, K_epochs)
    memory = Memory()
    
    max_episodes = 100
    max_timesteps = 8760 # One year of hourly data
    update_timestep = 2000 # Update policy every n timesteps
    
    timestep = 0
    
    # Training Loop
    for i_episode in range(1, max_episodes+1):
        obs_tuple = env.reset()
        # obs_tuple is (observations, info) in newer gym versions
        if isinstance(obs_tuple, tuple):
            state = obs_tuple[0]
        else:
            state = obs_tuple
            
        # Flatten state (concatenate all building observations)
        state = np.concatenate([np.array(s) for s in state])
        state = torch.FloatTensor(state).unsqueeze(0) # Add batch dim
        
        current_ep_reward = 0
        
        for t in range(max_timesteps):
            timestep += 1
            
            # Select action
            action, action_logprob = agent.policy_old.get_action(state)
            
            # Store data in memory
            memory.states.append(state.squeeze(0)) # Remove batch dim for storage
            memory.actions.append(action.squeeze(0))
            memory.logprobs.append(action_logprob.squeeze(0))
            
            # Execute action
            # Split action back into per-building actions
            action_np = action.numpy().flatten()
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
            
            # Sum reward if list (centralized reward)
            scalar_reward = sum(reward) if isinstance(reward, list) else reward
            
            memory.rewards.append(scalar_reward)
            memory.is_terminals.append(done)
            
            state = torch.FloatTensor(next_state).unsqueeze(0)
            current_ep_reward += scalar_reward
            
            # Update PPO agent
            if timestep % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                timestep = 0
            
            if done:
                break
        
        print(f"Episode {i_episode} \t Reward: {current_ep_reward:.2f} \t Steps: {t+1}")
        
        # Save model
        if i_episode % 10 == 0:
            torch.save(agent.policy.state_dict(), f'ppo_citylearn_{i_episode}.pth')
            print(f"Model saved at episode {i_episode}")

if __name__ == '__main__':
    main()
