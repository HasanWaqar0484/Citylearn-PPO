# PPO for Smart Grid Energy Management

**Production-grade implementation of Proximal Policy Optimization for building energy management using CityLearn.**

This project implements a corrected, publication-ready PPO agent addressing all critical issues identified in expert review.

## 🎯 Key Features

✅ **Correct PPO Implementation**
- Proper Generalized Advantage Estimation (GAE) with λ=0.95
- Clipped surrogate objective
- Value function clipping
- Gradient clipping
- KL divergence monitoring
- Entropy bonus scheduling

✅ **Production-Grade Infrastructure**
- Comprehensive logging (TensorBoard + file)
- Configuration management (YAML)
- Reproducibility (seeding, determinism)
- Modular architecture
- Extensive documentation

✅ **Baseline Comparisons**
- Random policy
- Do-nothing baseline
- Rule-based controller
- Statistical significance testing

✅ **Smart Grid Application**
- CityLearn environment integration
- 5 buildings with centralized control
- State normalization
- Episode statistics tracking

## 📁 Project Structure

```
ppo_smart_grid/
├── configs/                # Experiment configurations
│   └── default.yaml       # Default hyperparameters
├── src/
│   ├── agents/            # RL agents
│   │   ├── ppo.py        # PPO with proper GAE
│   │   └── baselines.py  # Baseline controllers
│   ├── envs/             # Environment wrappers
│   │   ├── citylearn_wrapper.py
│   │   └── normalization.py
│   ├── models/           # Neural networks
│   │   └── networks.py
│   └── utils/            # Utilities
│       ├── logger.py
│       ├── seeding.py
│       ├── metrics.py
│       └── config.py
├── experiments/          # Training/evaluation scripts
│   ├── train_ppo.py
│   └── evaluate.py
├── scripts/              # Utility scripts
│   └── debug_env.py
└── results/              # Outputs
    ├── checkpoints/
    ├── logs/
    └── plots/
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Or run directly
.\venv\Scripts\python -m pip install -r requirements.txt
```

### Training

```bash
# Train PPO agent
.\venv\Scripts\python experiments\train_ppo.py --config configs\default.yaml
```

### Evaluation

```bash
# Evaluate trained agent and compare with baselines
.\venv\Scripts\python experiments\evaluate.py --config configs\default.yaml
```

### Debugging

```bash
# Debug CityLearn environment structure
.\venv\Scripts\python scripts\debug_env.py
```

## 📊 What's Fixed

### Critical Fixes from Expert Review

1. **✅ Proper GAE Implementation**
   - Correct GAE(λ) formula from PPO paper
   - Proper bootstrapping for long episodes
   - Fixed advantage calculation

2. **✅ Correct Return Calculation**
   - No more vanishing gradients over 8760 steps
   - Proper value function bootstrapping
   - N-step returns with GAE

3. **✅ State Normalization**
   - Running mean/std normalization
   - Welford's online algorithm
   - Numerical stability

4. **✅ Comprehensive Logging**
   - TensorBoard integration
   - Training statistics (policy loss, value loss, KL, entropy)
   - Episode metrics
   - Gradient norms

5. **✅ Baseline Comparisons**
   - Random policy
   - Do-nothing baseline
   - Rule-based controller
   - Statistical aggregation

6. **✅ Reproducibility**
   - Seed setting for all RNGs
   - Deterministic algorithms
   - Configuration logging

## 🔬 Implementation Details

### PPO Algorithm

Based on [Schulman et al., 2017](https://arxiv.org/abs/1707.06347):

- **Clipped Objective**: `L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]`
- **GAE**: `Â_t = Σ(γλ)^l δ_{t+l}` where `δ_t = r_t + γV(s_{t+1}) - V(s_t)`
- **Value Loss**: `L^VF = (V_θ(s_t) - V^targ)^2`
- **Entropy Bonus**: `S[π_θ](s_t)`

### Hyperparameters

Default hyperparameters from `configs/default.yaml`:

- Learning rate: 3e-4
- Discount factor (γ): 0.99
- GAE lambda (λ): 0.95
- Clip range (ε): 0.2
- Entropy coefficient: 0.01
- Value function coefficient: 0.5
- Max gradient norm: 0.5
- Epochs per update: 10
- Batch size: 64
- Steps per update: 2048

## 📈 Expected Results

After training for 1M timesteps:

- **PPO should beat all baselines**
- **Reward improvement**: 20-40% over best baseline
- **Learning curve**: Gradual improvement over episodes
- **KL divergence**: Should stay below 0.02 (stable training)
- **Clip fraction**: Around 0.1-0.3 (good exploration)

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Make sure to run from project root
2. **CUDA errors**: Set `device: "cpu"` in config if no GPU
3. **Memory errors**: Reduce `batch_size` or `n_steps` in config
4. **Slow training**: Enable parallel environments (future feature)

### Debugging

Run the environment debugger to inspect CityLearn structure:

```bash
.\venv\Scripts\python scripts\debug_env.py
```

## 📝 Configuration

Edit `configs/default.yaml` to customize:

- Environment settings
- PPO hyperparameters
- Training parameters
- Logging options
- Experiment metadata

## 🔄 Next Steps

### Phase 2 Improvements (Optional)

1. **Offline Pre-training** (IQL/CQL)
2. **Safety Constraints**
3. **Parallel Environments**
4. **Hyperparameter Tuning**
5. **Ablation Studies**

See `implementation_plan.md` for full roadmap.

## 📚 References

1. Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
2. CityLearn Documentation: https://intelligent-environments-lab.github.io/CityLearn/
3. PPO Implementation Guide: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

## 🤝 Contributing

This is a research project. For improvements:

1. Follow existing code style
2. Add tests for new features
3. Update documentation
4. Run evaluation to verify changes

## 📄 License

MIT License - See LICENSE file for details

---

**Note**: This implementation addresses all critical issues from the expert review and provides a solid foundation for publication-quality research.
