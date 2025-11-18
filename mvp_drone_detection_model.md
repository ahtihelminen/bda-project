# MVP Bayesian Drone-vs-Background Detector

This document outlines a minimal, end-to-end Bayesian model for detecting drone presence in event camera data using binned event counts. It is designed as a starting point for a full Bayesian data analysis project.

---

## 1. Task Overview
We want to infer whether a drone is present in a given spatial patch of an event camera frame during a short time window.

- **Input:** Event counts in a patch over a time bin, e.g. Δt = 10 ms.
- **Output:** Posterior probability that a drone is present in that patch and time.

We assume access to annotated drone positions from the **Florence RGB-Event Drone dataset**.

---

## 2. Data Preparation
### 2.1 Temporal Binning
Divide events into fixed windows, e.g. 10 ms. For each bin $t$:
- Collect all events with timestamps in $[t, t+\Delta t)$.

### 2.2 Spatial Grid
Divide the event sensor into an $M \times N$ grid of patches.
For each patch $i$ and bin $t$:
- Count events → $y_{i,t}$.

### 2.3 Labels
Using the dataset’s drone bounding boxes:
- Label $z_{i,t} = 1$ if the drone intersects patch $i$ at any moment in bin $t$.
- Otherwise, $z_{i,t} = 0$.

This yields a table:
```
(sequence, patch_id, time_bin, y, z)
```

---

## 3. MVP Bayesian Model
### 3.1 Generative Assumption
Event counts differ depending on whether a drone is present. Use a Negative Binomial model to allow overdispersion.

For each observation $n$:
$
 y_n \mid z_n = c \sim \text{NegBinom}(\mu_c, \phi_c), \quad c \in \{0,1\}
$
where:
- $\mu_c$ is the mean count for class $c$
- $\phi_c$ controls dispersion

### 3.2 Priors
Weakly informative priors:
$
\log \mu_0, \log \mu_1 \sim \mathcal{N}(0, 2^2)
$
$
\log \phi_0, \log \phi_1 \sim \mathcal{N}(0, 1^2)
$

This forms a minimal non-hierarchical model.

---

## 4. Posterior Drone Probability
Given a new count $y$, compute:
$
P(z=1 \mid y, \text{data}) \propto P(y \mid z=1, \text{data}) \cdot P(z=1)
$
where:
- $P(z=1)$ is the empirical drone frequency.
- $P(y \mid z=c, \text{data})$ is the posterior predictive density.

Procedure:
1. Draw posterior samples of parameters.
2. For each sample, compute class likelihoods.
3. Average over samples.
4. Apply Bayes’ rule to get the posterior drone probability.

This produces the final detection signal.

---

## 5. Evaluation
Use a held-out sequence.
- Compute $P(z=1 \mid y)$ for each patch/time.
- Compare against ground truth labels.
- Measure ROC, precision–recall, and accuracy.
- Compare with a simple baseline: thresholding the raw count $y$.

---

## 6. Next Steps (for report extension)
After this MVP works:
- Add **hierarchical patch-level intercepts** to model spatial differences.
- Compare **Poisson vs Negative-Binomial** observation models.
- Incorporate simple **temporal/spatial covariates**.
- Perform **PPC**, **sensitivity analyses**, and **LOO model comparison**.

These extensions map directly to the course project requirements.

---

## Summary
This MVP provides:
- A clean binary drone detection task.
- A tractable Bayesian generative model.
- Straightforward inference using brms/Stan.
- Immediate pathways to richer hierarchical and comparative models.

It is simple enough to implement quickly but expandable enough for a full Bayesian data analysis workflow.

