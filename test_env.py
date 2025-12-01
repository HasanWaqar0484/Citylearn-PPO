import sys
from citylearn.citylearn import CityLearnEnv

print("Testing CityLearn environment...")

try:
    # Try to create environment with default dataset
    dataset_name = 'citylearn_challenge_2022_phase_1'
    print(f"Loading dataset: {dataset_name}")
    env = CityLearnEnv(dataset_name)
    print("Environment created successfully!")
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Try reset
    print("\nResetting environment...")
    obs = env.reset()
    print(f"Initial observation type: {type(obs)}")
    if isinstance(obs, list):
        print(f"Number of buildings: {len(obs)}")
        print(f"First building obs shape: {len(obs[0]) if isinstance(obs[0], list) else obs[0].shape}")
    
    print("\nEnvironment test successful!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
