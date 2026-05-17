# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & Commands

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
# Start JupyterLab (primary workspace)
uv run jupyter lab

# Install/sync dependencies
uv sync
```

The Python version is pinned in `.python-version` (3.12).

## Project Overview

RL assignment implementing **Imitation Learning** on `LunarLander-v3` (discrete actions, 8-dim state, 4 actions). All substantive work lives in `assignment1.ipynb`.

### Two-part structure

**Part A — Expert training:** Trains a PPO agent (via Stable-Baselines3) using 4 parallel envs, `EvalCallback` for checkpointing, targeting mean reward ≥ 200 over 100 episodes. Best checkpoint saved to `models/best_model.zip`.

**Part B — Behaviour cloning:** Treats imitation as 4-class supervised classification. Pipeline:
1. Collect `(state, action)` pairs from expert (200 episodes → ~70k transitions, saved to `data/demos_full.npz`)
2. Sub-sample into fixed-size datasets: `data/demos_{N}.npz` for N ∈ {100, 500, 1000, 5000, 10000, 50000, 20000, 80000}
3. Train `ImitationNet` (3-layer MLP: 8→64→64→4, cross-entropy, Adam) for each N; saved to `models/imitator_{N}.pt`
4. Evaluate imitator as greedy policy (argmax over logits) in live environment
5. Compare dataset sizes (B4) and data-selection strategies: random / coverage (farthest-point) / critical (high-entropy states) (B5)

### Key classes/functions in the notebook

- `ImitationNet` — the MLP classifier (cell `eb3f0c68`)
- `train_imitation()` — trains ImitationNet with 80/20 train-val split, best-val-loss model selection (cell `eb3f0c68`)
- `evaluate_imitator()` — runs ImitationNet as greedy policy, returns per-episode rewards and lengths (cell `23688a92`)
- `coverage_select()` / `critical_select()` / `random_select()` — few-shot data-selection strategies (cell `c29be3fd`)

### Artifacts

| Path | Contents |
|------|----------|
| `data/demos_full.npz` | Full 200-episode demo pool (states, actions) |
| `data/demos_{N}.npz` | Sub-sampled datasets |
| `models/best_model.zip` | SB3 PPO expert checkpoint |
| `models/imitator_{N}.pt` | ImitationNet state dicts |
| `logs/evaluations.npz` | SB3 EvalCallback log (timesteps, results) |
| `logs/*.png` | Assignment figures (training curves, comparisons) |
| `logs/video_expert/` | Recorded expert episode video |

## Fixed Hyperparameters

- `SEED = 42` throughout
- PPO: `N_ENVS=4`, `TOTAL_TIMESTEPS=5_000_000`, default SB3 hyperparameters
- ImitationNet training: 150 epochs, lr=1e-3, batch=256, val_frac=0.2
- Expert quality threshold: mean reward ≥ 200 over 100 episodes
