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
   - Use a **decaying trigger probability** \(p_{\text{trig}}(t)\) and a user-defined quota to control total extra evaluations.

---

## Features

- **Host-agnostic**: Works as a module on top of existing PSO / DE implementations.
- **Local, kernel-based geometry**: Uses RFF-based kernel ridge regression to approximate level sets and gradients.
- **Budget-aware**: At most a tiny fraction of extra function evaluations (e.g., ∼3%).
- **Monotone safeguard**: Never worsens the incumbent objective value in a KLSR call.
- **Simple integration**: One wrapper mechanism around your main optimization loop.

---
