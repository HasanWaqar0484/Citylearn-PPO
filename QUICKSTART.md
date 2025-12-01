# Quick Start Guide

## Activate Virtual Environment and Run Training

### Windows PowerShell:
```powershell
# Navigate to the project directory
cd C:\Users\H\.gemini\antigravity\scratch\Rl

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Run training
python train.py

# When done, deactivate
deactivate
```

### Windows Command Prompt:
```cmd
# Navigate to the project directory
cd C:\Users\H\.gemini\antigravity\scratch\Rl

# Activate the virtual environment
.\venv\Scripts\activate.bat

# Run training
python train.py

# When done, deactivate
deactivate
```

## Alternative: Run Without Activating

You can also run Python directly from the virtual environment without activating:

```powershell
.\venv\Scripts\python train.py
```

## Files Overview

- `train.py` - Main training script (trains for 100 episodes)
- `evaluate.py` - Evaluation script (tests trained model)
- `ppo_agent.py` - PPO algorithm implementation
- `env_wrapper.py` - CityLearn environment wrapper
- `test_env.py` - Quick environment test

## Expected Output

Training will show:
```
Observation Space: [Box(...), Box(...), ...]
Action Space: [Box(...), Box(...), ...]
Number of buildings: 5
Total State Dim: 140, Total Action Dim: 5
Episode 1    Reward: -XXXX.XX    Steps: XXXX
Episode 2    Reward: -XXXX.XX    Steps: XXXX
...
Model saved at episode 10
```

The training takes time as it loads the CityLearn dataset and runs through episodes.
