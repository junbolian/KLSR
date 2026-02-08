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

