# Assignment 1 — Implementation Review

## Part A — Training an Expert RL Agent

**Status: Correct overall.**

All four requirements are met: SB3/PPO is used, the agent improves and reaches ≥ 200, a training curve is plotted, and the performance criterion is stated.

### Issue: Stale expert evaluation text
The markdown in the Expert Evaluation cell (`1ee4a3e8`) says:
> *"mean episodic reward of 173.09 ± 99.68… falls below the expert threshold of 200"*

But the actual cell output directly below shows **242.90 ± 38.78**, which is above the threshold. The written analysis was never updated after re-training. The same stale numbers appear again in the B3 overview note (*"~163–194 mean reward"*) — a direct contradiction between the written analysis and the results.

---

## Part B — Imitation Learning

### B1 — Collecting Demonstration Data
Mostly correct, but:

**Markdown table vs. code mismatch:** The B1 description table lists sizes `{100, 500, 1000, 5000, 10000, 50000}`, but `DATASET_SIZES` in the code is `[100, 1000, 5000, 20000, 80000]`. Sizes 500 and 10,000 are never generated; 20,000 and 80,000 are generated but not mentioned in the table. Graders will see the discrepancy.

### B2 — Training an Imitation Model
✓ Correct.

### B3 — Evaluating the Imitation Model
✓ Correct.

### B4 — Effect of Dataset Size
✓ Correct — all three required plots are present (reward, success rate, val accuracy vs. dataset size).

### B5 — Few-Shot Imitation Learning
✓ Correct logic. Make sure the quantity-vs-quality answer in the discussion cell is also included explicitly in the PDF report.

---

## Cross-Cutting Issues

| # | Issue | Severity |
|---|-------|----------|
| 1 | **Author names not filled in** — cell 0 still says `*(fill in names and IDs)*` | Critical |
| 2 | **Stale expert evaluation text** — written analysis says 173.09 / below-threshold, but output shows 242.90 / above | Critical |
| 3 | **No imitator video** — submission guidelines require video clips showing relevant progress after convergence; only `logs/video_expert/expert-episode-0.mp4` exists. No recorded episode of the imitator policy. The TODO comment in the video cell (`# TODO: what should be recorded as video?`) is unresolved | High |
| 4 | **Four TODO comments left in code** — `# TODO: check with the lecturer about what policy to use`, `# TODO: change this to not ai print`, `# TODO: consider change the sizes`, `# TODO: what should be recorded as video?`. These will hurt the "clean code" grade criterion explicitly called out in the submission guidelines | Medium |
| 5 | **B1 dataset table doesn't match actual sizes** — see B1 note above | Medium |

---

## What to Fix Before Submission

1. Fill in author names/IDs in cell 0.
2. Update the Expert Evaluation markdown to reflect the actual result (242.90 ± 38.78, criterion met).
3. Update the B3 overview note to remove the stale ~163–194 claim.
4. Add a video recording cell for the best imitator (mirror the expert video cell) and display it inline — required by submission guidelines.
5. Remove all four TODO comments.
6. Correct the B1 markdown table to match `DATASET_SIZES = [100, 1000, 5000, 20000, 80000]`.
