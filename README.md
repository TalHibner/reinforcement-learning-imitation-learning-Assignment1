# Reinforcement Learning: Imitation Learning Assignment

## Overview
This assignment implements **Imitation Learning** (Behavior Cloning) on the `LunarLander-v3` environment. The task is to train an agent to mimic an expert policy by learning from its demonstrations.

## Project Structure

### Part A: Expert Training
- Train a PPO agent using Stable-Baselines3
- Use 4 parallel environments for efficient training
- Target mean reward ≥ 200 over 100 episodes
- Best model saved to `models/best_model.zip`

### Part B: Behavior Cloning
- Collect state-action pairs from expert (200 episodes → ~70k transitions)
- Train `ImitationNet` (3-layer MLP: 8→64→64→4) as a 4-class classifier
- Evaluate with different dataset sizes: 100, 500, 1000, 5000, 10000, 20000, 50000, 70000
- Compare data-selection strategies: random, coverage (farthest-point), critical (high-entropy states)

## Files & Artifacts

| Path | Contents |
|------|----------|
| `assignment1.ipynb` | Main notebook with complete implementation |
| `data/demos_full.npz` | Full 200-episode demonstration pool |
| `data/demos_{N}.npz` | Sub-sampled datasets |
| `models/best_model.zip` | SB3 PPO expert checkpoint |
| `models/imitator_{N}.pt` | Imitation network weights |
| `logs/` | Training curves, evaluation results, and episode videos |

## Setup & Usage

```bash
# Install dependencies (uses uv)
uv sync

# Start JupyterLab
uv run jupyter lab
```

Then open `assignment1.ipynb` to run the full pipeline.

## Environment Details
- **Environment:** LunarLander-v3 (discrete actions: 0-3, 8-dim state)
- **Python:** 3.12
- **Key Dependencies:** PyTorch, Stable-Baselines3, Gymnasium

## Hyperparameters
- SEED: 42
- PPO: N_ENVS=4, TOTAL_TIMESTEPS=5M
- ImitationNet: 150 epochs, lr=1e-3, batch_size=256, val_split=0.2
