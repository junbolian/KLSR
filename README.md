# KLSR: Kernel Level–Set Reflection for Metaheuristic Optimization

This repository implements **Kernel Level–Set Reflection (KLSR)**, a lightweight, budget-aware reflection operator that plugs into population-based metaheuristics (e.g., PSO, DE) for **continuous black-box optimization**.

KLSR does **not** replace your optimizer. Instead, it acts as a **host-agnostic module** that:
- Detects **stalled** individuals,
- Builds a **local surrogate** around the global best using kernel ridge regression with random Fourier features (RFF),
- Uses the surrogate’s gradient to define a **local level-set mirror**, and
- Performs **reflection moves** that are only accepted if they strictly improve the objective.

The goal: **better solutions and faster progress** with only a **small increase** in evaluation budget.

---

## Method Overview

### Problem Setting

We consider bound-constrained black-box minimization:

$$
\min_{x \in \mathbb{R}^d} f(x) \quad \text{s.t.} \quad \ell \le x \le u
$$

with expensive evaluations and no exact gradients.

Population-based metaheuristics (PSO, DE, etc.) maintain:
- Current position \(x\),
- Personal best \(p\),
- Global best \(g\).

KLSR wraps around these methods and occasionally proposes **reflected candidates** for selected individuals.

---

### Core Idea: Kernel Level–Set Reflection

1. **Local surrogate around the global best**  
   - Maintain an archive of evaluated points $(X_i, F_i)$.
   - Rescale to $[0,1]^d$ and select $k$ nearest neighbors around the current global best $g$.
   - Fit a **kernel ridge regressor** using **random Fourier features (RFF)** to approximate a Gaussian kernel smoother.

2. **Learned mirror and gradient**  
   - The surrogate yields an approximate gradient $\nabla \hat f(g)$.
   - This defines a hyperplane that locally approximates a **level set**:
     - Center: $c = g$
     - Normal: $n = \nabla \hat f(g) / \|\nabla \hat f(g)\|$

3. **Specular reflection moves**  
   For a stalled individual $x$, reflection across the learned mirror:

$$
y(\alpha) = x - 2\alpha\, \langle x - c, n \rangle\, n,\quad \alpha \in \{1, 0.5, 0.25\}.
$$

   - Apply **bound handling** (`clip` or `mirror`) to keep $y$ in $[\ell, u]$.
   - If the surrogate is unreliable (low $R^2$ or tiny gradient), fall back to simple symmetry reflections around $p$ and $g$:

$$
y_p = 2p - x,\quad y_g = 2g - x,\quad y_{pg} = 2g - (2p - x).
$$

5. **Monotone, budget-aware adoption**
   - For each KLSR call, evaluate at most a **small number** of candidates (typically one).
   - Only accept a candidate if it **strictly improves** the objective.
   - This makes each KLSR invocation **non-degrading** and evaluation overhead **negligible** (a few percent of baseline).

6. **Stall-aware triggering mechanism**
   - Track **stall length** for each individual and for the global best.
   - Periodically refit the mirror when the global best has stalled for enough iterations.
   - Apply KLSR only to a small subset of the **most stagnated** individuals.
   - Use a **decaying trigger probability** $p_{\text{trig}}(t)$ and a user-defined quota to control total extra evaluations.

---

## Features

- **Host-agnostic**: Works as a module on top of existing PSO / DE implementations.
- **Local, kernel-based geometry**: Uses RFF-based kernel ridge regression to approximate level sets and gradients.
- **Budget-aware**: At most a tiny fraction of extra function evaluations (e.g., ∼3%).
- **Monotone safeguard**: Never worsens the incumbent objective value in a KLSR call.
- **Simple integration**: One wrapper mechanism around your main optimization loop.

---


----------------------------------------
A) Statistics / Testing / Summaries
----------------------------------------
Cal_stats.m
- Aggregates per-algorithm performance across runs.
- Produces summary stats (best/worst/mean/std + mean time).
- Performs nonparametric significance tests vs baseline:
  - signrank (paired)
  - ranksum  (unpaired)
- Runs Friedman test across all algorithms.

Cal_stats_dif.m
- Same idea as Cal_stats, but compares algorithms in fixed PAIRS:
  (1 vs 2), (3 vs 4), (5 vs 6)
- Useful when your results matrix is arranged as:
  [AlgoA_base, AlgoA_plus, AlgoB_base, AlgoB_plus, ...].

----------------------------------------
B) Random Distributions (used by JADE/SHADE)
----------------------------------------
cauchyrnd.m
- Generates random samples from a Cauchy distribution.
- Commonly used for sampling DE scale factor F in JADE/SHADE.

cauchyinv.m
- Inverse CDF (quantile) for Cauchy distribution.
- Used internally by cauchyrnd or parameter sampling logic.

----------------------------------------
C) CEC Benchmark Backends
----------------------------------------
cec17_func.cpp
- CEC2017 objective function implementation (MEX source).
- Must be compiled to call from MATLAB as: cec17_func(x, func_id).

cec22_func.cpp
- CEC2022 objective function implementation (MEX source).
- Must be compiled to call from MATLAB as: cec22_func(x, func_id).


----------------------------------------
D) Benchmark Wrapper Utilities
----------------------------------------
Get_Functions_cec2017.m
- Returns lb/ub and objective handle fobj for CEC2017 function F=1..30.
- Typical bounds: [-100,100] per dimension.
- fobj = @(x) cec17_func(x', F);  (note transpose: column vector)

Get_Functions_cec2022.m
- Returns lb/ub and objective handle fobj for CEC2022 function F=1..12.
- Typical bounds: [-100,100], except some functions (e.g., F11) use [-600,600].
- fobj = @(x) cec22_func(x', F);

func_plot_cec2017.m
- Plots a 3D surface of a CEC2017 function using surfc().
- If dim>2, it plots the slice where x3..xd = 0.

func_plot_cec2022.m
- Same as above for CEC2022.

----------------------------------------
E) Initialization Helper
----------------------------------------
initialization.m
- Creates an N×dim initial population uniformly within bounds.

----------------------------------------
F) Baseline Optimizers
----------------------------------------
PSO.m
- Particle Swarm Optimization baseline.
- Returns best score, best position, and convergence curve.

DE.m
- Differential Evolution baseline.
- Returns best score, best position, and convergence curve.

JADE.m
- JADE (adaptive DE):
  - Uses Cauchy/Normal sampling for F/CR around evolving means.
  - DE/current-to-pbest style mutation (typical JADE design).
- Depends on cauchyrnd/cauchyinv for F sampling.

SHADE.m
- SHADE (Success-History Adaptive DE):
  - Maintains memory arrays for CR and F (success-history).
  - Uses archive and weighted updates based on improvements.

----------------------------------------
G) KLSR (Kernel Level–Set Reflection)
----------------------------------------
KLSR.m
- The KLSR operator itself (reflection move logic).
- Conceptually:
  - Detect “stalled” individuals.
  - Fit local surrogate around (usually) current best or a center.
  - Use gradient direction as a “mirror normal”.
  - Reflect candidate(s) and accept only if improvement.
  - Handles bounds via clip or mirror strategy (depending on your options).

KLSR_fitModel.m
- Fits a local surrogate model used by KLSR.
- Uses Random Fourier Features + ridge regression (fast kernel approximation).
- Outputs:
  - reliability (e.g., R^2 / quality)
  - gradient estimate
  - normal vector used for reflection

PSO_KLSR.m
- PSO + KLSR integration wrapper.
- Runs PSO loop, and periodically triggers KLSR reflections per rules.

DE_KLSR.m
- DE + KLSR integration wrapper.
- Runs DE loop, and periodically triggers KLSR reflections per rules.

----------------------------------------
H) CCE (Competitive Cluster Elimination)
----------------------------------------
(main scripts reference CCE.m, and your folder contains it)
CCE.m
- Clusters the population (k-means style).
- Identifies the “worst” cluster using a rule (min/median).
- Replaces a fraction rho of the cluster’s worst members by reinitialization
  or other reset strategy.
- Controlled by:
  - tau   : refresh interval (how often clustering happens)
  - rho   : fraction replaced within the worst cluster
  - K     : number of clusters (or auto choice)
  - steps : k-means inner iterations
  - keep_one / avoid_best : safety options

----------------------------------------
I) Experiment Drivers (the “main_*.m” entry points)
----------------------------------------
main_CCE_compare.m
- Baseline (PSO or DE) vs CCE-enhanced version on CEC2017 (F1..F30).
- Grid search over (tau, rho); each combo runs multiple trials.
- Chooses best (tau,rho) by median best fitness.
- Saves:
  - Convergence plot, boxplot
  - 3D bar + surface plots over tau×rho
  - results_*.mat with all cached data

main_CCE_generalize_fixed.m
- Uses fixed CCE parameters (global CCE_OPTS) across multiple algorithms:
  e.g., DE, SHADE, JADE (baseline vs CCE version).
- Runs multiple trials per function, generates:
  - Per-function convergence plots per algorithm
  - Combined boxplots
  - LaTeX tables + win/tie/loss summary (Wilcoxon rank-sum)

main_KLSR_compare.m
- Baseline (PSO or DE) vs KLSR-enhanced version on CEC2017.
- Grid search over (tau, p0) only (two most important KLSR params).
- Saves:
  - Convergence plot, boxplot
  - 3D bar + surface plots over tau×p0
  - results_*.mat

----------------------------------------
J) Paper/Post-processing Tools
----------------------------------------
make_cce_paper_tables.m
- Post-processes results_<CCE_...>.mat from main_CCE_compare.
- Computes average ranks across functions, explicitly including baseline.
- Exports:
  - Avg-rank plots (3D bar/surface)
  - CSV + LaTeX for avg-rank
  - Per-function best config summary CSV + LaTeX longtable
  - Comprehensive Excel with everything needed for paper tables

plot_avg_rank3D.m
- Alternate rank plotting/export tool:
  - Shows baseline as a single bar
  - Shows the tau×rho grid bars separately
- Exports Rank.png alias + PDF + CSV + LaTeX + Excel.