# Reinforcement Learning – Imitation Learning – Assignment 1

## Overview

This assignment implements **Imitation Learning** (Behavior Cloning) on `LunarLander-v3`. The project trains a PPO expert policy, collects demonstrations, and trains neural networks to mimic the expert using supervised learning from varying dataset sizes.

**See [Reinforcement Learning – Imitation Learning – Assignment1.pdf](Reinforcement%20Learning%20–%20Imitation%20Learning%20–%20Assignment1.pdf) for full methodology, results, and analysis.**

---

## Quick Start

### Setup
```bash
# Requires Python 3.12 (pinned in .python-version)
uv sync
uv run jupyter lab
```

All work is in `assignment1.ipynb`.

---

## Project Structure

```
data/
  demos_full.npz         Full 70k expert demonstrations
  demos_{N}.npz          Sub-sampled datasets (N ∈ {100, 500, ..., 70000})

models/
  best_model.zip         Expert policy (PPO, 5M timesteps)
  imitator_{N}.pt        Imitation networks by dataset size

logs/
  evaluations.npz        Expert evaluation metrics
  *.png                  Training curves and comparisons
  videos/                Episode recordings

assignment1.ipynb        Main notebook with all code
```

---

## Setup & Installation

### Prerequisites
- Python 3.12+
- PyTorch, Gymnasium, Stable-Baselines3

### Installation

```bash
# Clone repository
git clone https://github.com/TalHibner/reinforcement-learning-imitation-learning-Assignment1.git
cd reinforcement-learning-imitation-learning-Assignment1

# Install & run (using uv)
uv sync
uv run jupyter lab

# Or with pip
pip install -r requirements.txt
jupyter lab
```

### Running

```bash
jupyter lab assignment1.ipynb
```

---

## Key Concepts

### PPO (Proximal Policy Optimization)
- On-policy policy gradient algorithm
- Clipped probability ratios prevent large destabilizing updates
- Balances sample efficiency and stability

### Behavior Cloning
- Learn policy as supervised classification
- Simple and fast to train
- Suffers from distribution shift and error compounding

---

## Results Summary

### Part A ✅
- Mean reward ≥ 200 achieved
- 95-100% episode success rate
- Convergence by 1-2M timesteps

### Part B ✅
- ImitationNet: ~95% validation accuracy
- Expert-level performance replicated
- 50k+ samples near-optimal

---

## Authors

- **Gilad Ticher** (ID: 318770039)
- **Tal Hibner** (ID: 026548446)

---

## References

1. Schulman et al. (2017). "Proximal Policy Optimization Algorithms"
2. Bain & Sammut (1995). "A Framework for Behavioral Cloning"
3. Hussein et al. (2017). "Imitation Learning: A Bibliographic Survey"
4. Stable-Baselines3: https://stable-baselines3.readthedocs.io/
5. Gymnasium: https://gymnasium.farama.org/
