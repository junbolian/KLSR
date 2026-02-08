# Kernel Level–Set Reflection (KLSR)

This repository provides the reference implementation of **Kernel Level–Set Reflection (KLSR)**, a lightweight operator for evolutionary and swarm-based black-box optimization under fixed evaluation budgets. KLSR augments a host optimizer by learning a local “mirror” map from previously evaluated samples via random-feature ridge regression and then proposing conservative reflection steps subject to a monotone acceptance safeguard. The operator is activated through a stall-aware gate and is implemented without altering the host algorithm’s native update rules.

<p align="center">
  <img src="klsr_workflow.png" alt="KLSR workflow" width="700">
</p>

**Key contributions**:

- A geometry-aware reflection operator is proposed for evolutionary algorithms.  
- Local surrogate gradients define adaptive reflection hyperplanes.  
- Monotone acceptance and stall-aware triggering ensure safe integration.  
- Four metaheuristics are consistently improved on CEC2017 benchmarks.  
- Validated on UAV planning, SVM tuning, and feature selection tasks.

---

## Scope and Capabilities

KLSR is designed as a plug-in mechanism that preserves standard fixed-budget comparison protocols:

- **Budget discipline**: the total number of objective evaluations is unchanged.  
- **Stagnation-aware invocation**: reflections are proposed only when global progress plateaus.  
- **Host-agnostic integration**: KLSR operates as an add-on module rather than a redesign of baseline dynamics.  
- **Applicability across encodings**: both continuous and binary decision-variable settings are included.  

---

## Implemented Host Optimizers

Baseline implementations and their KLSR-augmented counterparts are provided for:

- Competitive Swarm Optimizer (CSO)  
- Genetic Algorithm (GA)  
- JADE (adaptive Differential Evolution)  
- Particle Swarm Optimization (PSO)  

---

## How to Run (Quick Guide)

The main entry points are:

- `main_KLSR_master.m`: primary runner. You can run a single algorithm or all algorithms in one go; choose dimensions; switch modes; and optionally save plots.
- `main_KLSR_compare.m`: legacy/backup script (earlier pipeline). It is kept for reference and consistency but is not required for typical runs.

### 1. Prerequisites

- MATLAB (64-bit).
- CEC2017 function available:
  - Windows: `cec17_func.mexw64` is already included.
  - Other OS: compile `cec17_func.cpp` using `mex`.
- Data folder `input_data17/` must exist in the repo (CEC2017 data).

### 2. Open MATLAB and set path

```matlab
cd('d:\paper\KLSR');   % or your repo path
addpath(genpath(pwd));
```

### 3. Run the master script

Edit options at the top of `main_KLSR_master.m` (single vs all algorithms, dims, mode, plot switch, etc.), then run:

```matlab
main_KLSR_master
```

Outputs are saved to the current working directory:

```
results_KLSR_<ALGO>_CEC2017_D<dim>_<mode>.mat
```

Example:

```
results_KLSR_PSO_CEC2017_D30_sig.mat
```

### 4. Notes

- Results are saved in the repo root unless you change the save path in the scripts.
- `main_KLSR_master.m` supports: single-algo or all-algo runs, plot on/off, and different modes.
- Random seeds are fixed in `main_KLSR_master.m` (`master_seed`), so results are reproducible.

### 5. Running S1–S10 (Sensitivity + Comparison Pipeline)

Typical order to run (adjust if you only need part of the pipeline):

- S1-S5: Sensitivity Analysis.
- S6-S8: Compariosn.
- S9-S10: Statistics

---

## File Map (What Each Part Does)

Core entry points:

- `main_KLSR_master.m`: primary runner (baseline vs KLSR), supports single/all algorithms, modes, plots.
- `main_KLSR_compare.m`: legacy/backup comparison pipeline.

Algorithms:

- `CSO.m`, `GA.m`, `JADE.m`, `PSO.m`: baseline optimizers.
- `CSO_KLSR.m`, `GA_KLSR.m`, `JADE_KLSR.m`, `PSO_KLSR.m`: KLSR-augmented versions.
- `KLSR.m`, `KLSR_fitModel.m`: KLSR operator + local surrogate fitting.

CEC2017 benchmark:

- `Get_Functions_cec2017.m`: function handles + bounds for CEC2017.
- `cec17_func.mexw64`: Windows 64-bit compiled function.
- `cec17_func.cpp`: source for compiling on other OS.
- `input_data17/`: CEC2017 data files (required).

Sensitivity / comparison pipeline scripts:

- `S1_run_grid_collect_raw.m`: run grid search and collect raw results.
- `S2_summarize_table_sens.m`: summarize sensitivity table.
- `S3_plot_avg_rank_layout.m`: average rank plots.
- `S4_plot_boxplot_grid_layout.m`: boxplot grid plots.
- `S5_export_Table_sens.m`: export sensitivity table.
- `S6_run_compare_collect_raw.m`: run comparison experiments.
- `S7_summarize_compare_table.m`: summarize comparison table.
- `S8_export_table_compare.m`: export comparison table.
- `S9_stats_friedman_compare.m`: Friedman test for comparison.
- `S10_stats_cliffs_compare.m`: Cliff’s delta effect size stats.

