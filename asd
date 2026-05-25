uv run jupyter nbconvert --to latex --no-input assignment1.ipynb && \
sed -i '' 's/\\def\\LTcaptype{none}/\\def\\LTcaptype{table}\\addtocounter{table}{-1}/g' assignment1.tex && \
xelatex -interaction=nonstopmode assignment1.tex && \
xelatex -interaction=nonstopmode assignment1.tex


uv run jupyter nbconvert --to latex --no-input assignment1.ipynb && \
sed -i '' \
  's/\\def\\LTcaptype{none}/\\def\\LTcaptype{table}\\addtocounter{table}{-1}/g' \
  assignment1.tex && \
sed -i '' '/\\title{/d; /\\maketitle/d; /\\author{/d; /\\date{/d' \
  assignment1.tex && \
sed -i '' 's/\\begin{document}/\\begin{document}\\newgeometry{margin=1.5cm}/' \
  assignment1.tex && \
xelatex -interaction=nonstopmode assignment1.tex

uv run jupyter nbconvert --to latex --no-input assignment1.ipynb && \
sed -i '' \
  's/\\def\\LTcaptype{none}/\\def\\LTcaptype{table}\\addtocounter{table}{-1}/g' \
  assignment1.tex && \
sed -i '' '/\\title{/d; /\\maketitle/d; /\\author{/d; /\\date{/d' \
  assignment1.tex && \
sed -i '' '/{ \\hspace\*{\\fill} \\\\}/d' \
  assignment1.tex && \
sed -i '' 's/\\begin{document}/\\begin{document}\\newgeometry{margin=1.5cm}\\setlength{\\floatsep}{2pt}\\setlength{\\textfloatsep}{2pt}\\setlength{\\intextsep}{2pt}/' \
  assignment1.tex && \
xelatex -interaction=nonstopmode assignment1.tex



**How examples were selected:**

Five strategies were compared across dataset sizes of 100, 500, 750, and 1,000 samples (5 independent runs each):

- *Random*: uniform sample from the 200-episode pool — no selection bias, baseline.

- *Coverage*: greedy farthest-point sampling in normalised state space — maximally spreads selected states across the 8-dimensional observation space so the imitator sees representative states from all regions.

- *Critical*: selects the top-N states ranked by expert action entropy — the most ambiguous decision-boundary states where the expert is least confident.

- *Stratified Coverage*: farthest-point sampling run **per action class** with proportional class budgets (derived from the pool's action distribution), then merged. Ensures every action type is represented while preserving intra-class spatial diversity via PCA-projected state features.

- *Density-Reweighted Coverage*: farthest-point sampling restricted to the **dense core** of the pool. States in the top 5% of kNN distance (the isolated outlier fringe) are discarded before running coverage, so the budget is not wasted on rare transitions that are unlikely to recur during online rollouts.

**What worked well:** Plain Coverage was the strongest strategy across all dataset sizes. By greedily maximising spread in normalised state space, it ensures the imitator is exposed to qualitatively different flight conditions even with very few samples — reaching $\sim$ 233 mean reward at N=500 while every other strategy was still negative or marginal. Density-Reweighted Coverage was the second-best approach: by pruning outlier states before running coverage it slightly narrows the gap at larger N (N=750, N=1000), where both strategies converge to $\sim$ 200–210.

**What did not work well:** Critical (entropy-based) selection was the worst strategy at every dataset size, consistently achieving strongly negative mean rewards (below −450 even at N=1000). Concentrating all training examples near the expert's decision boundaries starves the imitator of the confident, high-frequency manoeuvres — sustained main-engine thrust and basic orientation control — that constitute the bulk of a successful episode. Stratified Coverage underperformed at small N (100–500): enforcing per-class budgets via a within-class farthest-point search in PCA space restricts each class to a narrow intra-class manifold, losing the global diversity that plain coverage captures. It recovered to competitive performance only at N $\geq$ 750, once the per-class budgets were large enough to span the full class geometry. Random sampling improved steadily with N but remained poor at very low N because a small uniform draw has high variance and can miss entire regions of the state space by chance.

**Answer — data quantity or quality?**

At the scale tested, **quality matters more at low N**: with only 100–500 samples, deliberate coverage-based selection dramatically outperforms random sampling. Plain Coverage achieves positive mean reward ($\sim$ 233) at N=500 while random still averages around −300. Past $\sim$ 1,000 samples the advantage narrows — a larger random dataset naturally explores sufficient state-space diversity — but even at N=1000, Coverage and Density-Reweighted Coverage remain noticeably ahead of random.



**Authors:** Gilad Ticher 318770039, Tal Hibner 026548446