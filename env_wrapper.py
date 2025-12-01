import gymnasium as gym
import numpy as np
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

class CustomCityLearnWrapper(gym.Wrapper):
    def __init__(self, env):
        # Use CityLearn's built-in wrappers for normalization and SB3 compatibility if available
        # But here we wrap manually to ensure we control the reward
        super().__init__(env)
        self.env = env
        
        # Define weights for the multi-objective reward
        # Weights can be adjusted based on preference or specific assignment details if given
        self.weights = {
            'electricity_consumption': 1.0,
            'carbon_emissions': 1.0,
            'electricity_price': 1.0,
            'discomfort': 1.0
        }

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Custom Multi-Objective Reward Calculation
        # We extract metrics from info. 
        # Note: The keys might vary based on CityLearn version.
        # Common keys: 'electricity_consumption', 'carbon_emissions', 'pricing', 'net_electricity_consumption'
        
        # Initialize components
        cost = 0.0
        carbon = 0.0
        peak = 0.0
        
        # Try to extract from info if available (usually a list of dicts for each building)
        if isinstance(info, list) and len(info) > 0:
            # Aggregate over buildings
            for building_info in info:
                cost += building_info.get('electricity_consumption', 0) * building_info.get('electricity_price', 1.0)
                carbon += building_info.get('carbon_emissions', 0)
                # Peak is harder to calculate per step, usually it's over a window or episode.
                # Here we penalize high consumption to approximate peak reduction.
                peak += max(0, building_info.get('net_electricity_consumption', 0))
        elif isinstance(info, dict):
             cost += info.get('electricity_consumption', 0) * info.get('electricity_price', 1.0)
             carbon += info.get('carbon_emissions', 0)
             peak += max(0, info.get('net_electricity_consumption', 0))
             
        # Normalize or scale components
        # We want to minimize these, so reward is negative
        
        reward_val = - (self.weights['electricity_consumption'] * cost + 
                        self.weights['carbon_emissions'] * carbon + 
                        self.weights['discomfort'] * peak) # Using peak as discomfort/grid stress proxy
                        
        # If the environment returns a reward, we can combine it or replace it.
        # For now, let's mix them to ensure we don't lose environment specific signals (like comfort)
        
        # If reward is a list (one per building), sum it up for centralized training
        original_reward = sum(reward) if isinstance(reward, list) else reward
        
        final_reward = original_reward + reward_val * 0.1 # Weighting our custom component
        
        return obs, final_reward, done, truncated, info
