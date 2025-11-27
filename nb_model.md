# Simple Negative Binomial Classifier

This document describes the minimal event-count–based classifier used throughout the project. It mirrors the structure and tone of the main project README.

---

## Introduction

Event cameras output streams of brightness-change events rather than traditional frames. When a drone moves across the field of view, it produces short, dense bursts of events. Background regions, by contrast, generate only sparse activity.

By dividing the sensor into spatial patches and aggregating events into short temporal windows, we obtain simple integer counts. These counts differ systematically depending on whether a drone is present. The Negative Binomial (NB) classifier leverages this fact by modeling the event count distribution under two competing hypotheses:

- **Background (z = 0)**
- **Drone present (z = 1)**

This yields a lightweight Bayesian detector suitable for real-time inference.

---

## Data Construction

### Temporal Binning
Events are grouped into windows of fixed duration (e.g., 10–50 ms). For each window $t$:

$
 y_{i,t} = \text{\#events in patch } i \text{ during } t
$

### Spatial Grid
The sensor is divided into an $M \times N$ grid. Each patch produces one count per time window.

### Labels
Ground-truth bounding boxes (interpolated when necessary) are mapped onto the grid. A patch is labeled drone-positive if the drone’s box intersects the patch at any time during the window.

This produces a table of rows of the form:
```
(sequence_id, patch_id, time_bin, y, z)
```

---

## 3. Model

The core assumption is that event counts follow different overdispersed distributions based on class membership:

$
 y_n \mid z_n = c \sim \text{NegBinom}(\mu_c,\;\phi_c),
 \quad c \in \{0,1\}
$

- $\mu_c$: expected event count for class $c$
- $\phi_c$: dispersion controlling how heavy-tailed the distribution is

This accommodates the high variance seen in background activity and the extreme bursts during drone motion.

## 4. Priors

We place weakly informative priors on all parameters to ensure the model is identifiable, numerically stable, and regularized without being overly restrictive. For each class $c \in \{0,1\}$, we define:

$
\log \mu_c \sim \mathcal{N}(0,\,2^2), \qquad 
\log \phi_c \sim \mathcal{N}(0,\,1^2)
$

These choices serve several purposes:

1. **They enforce proper Bayesian inference.**  
   All parameters have fully proper priors, satisfying project requirements and ensuring the posterior is well-defined.

2. **They incorporate domain-appropriate scale information without constraining the model too tightly.**  
   - The count means $\mu_c$ are expected to range from “a fraction of an event per window” to “tens of events per window.”  
   - A Normal(0, 2) prior on the log-scale corresponds roughly to allowing $\mu_c$ to vary over multiple orders of magnitude (from ~0.02 to ~50) while mildly discouraging unrealistic extremes.  
   This reflects what we know about event-camera counts: background is often near zero, and drone patches can spike dramatically.

3. **They stabilize inference.**  
   The dispersion parameters $\phi_c$ strongly affect the shape of the NB distribution.  
   A Normal(0, 1) prior on $\log \phi_c$ constrains $\phi_c$ to a reasonable region (roughly 0.1–10 on the original scale), preventing numerical issues while still allowing substantial overdispersion.

4. **They avoid unintentionally encoding strong beliefs.**  
   Since we want the data to drive the class separation, these priors regularize but do not dominate. They express weak expectations about scale while letting the model learn the true differences between background and drone counts.

Overall, these priors strike the intended balance:  
- strong enough to regularize inference and prevent degeneracy,  
- weak enough to allow the highly variable event-count data to inform the posterior.


## 5. Stan code

From the repo check models/nb.stan

## 6. MCMC Inference

We fit the simple 2-class NB model using Stan (via cmdstanpy) on a random subset of the labeled counts table. Conceptually, the fitting step looks like this:

    model = CmdStanModel(stan_file="nb.stan")
    fit = model.sample(
        data={
          "N": len(y),
          "y": y,
          "z": z,
        },
        chains=4,
        parallel_chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        seed=1337,
    )

and we use the following configuration and rationale:

- Data subsampling:
  - If the full table has more than max_samples = 30 000 rows, we draw a random subset of exactly 30 000 rows (with a fixed seed for reproducibility).
  - This keeps computation time manageable while still giving enough information to estimate the four global parameters (mu0, mu1, phi0, phi1).

- Number of chains and iterations:
  - chains = 4 parallel Markov chains.
  - iter_warmup = tune = 1000 warmup iterations per chain.
  - iter_sampling = draws = 1000 post-warmup iterations per chain.
  - This yields up to 4000 posterior draws per parameter, which is typically sufficient for good effective sample sizes and accurate uncertainty summaries for this low-dimensional model.

- Parallelization:
  - parallel_chains = cores, with cores = min(chains, 4).
  - This uses up to 4 CPU cores, so all chains can run in parallel on a typical laptop without overcommitting resources.

- Random seed:
  - seed = random_state = 1337.
  - A fixed seed makes the sampling run reproducible, which is important for debugging, comparison between model versions, and for the project report.

- Saved outputs:
  - The cmdstanpy fit is converted to an ArviZ InferenceData object (including the pointwise log-likelihood log_lik from the Stan generated quantities block) and saved to disk as nb_idata.nc.
  - This stored object is then reused for:
    - convergence diagnostics (part 7),
    - posterior predictive checks (part 8),
    - predictive performance and LOO-CV (part 9 and 11),
    - and sensitivity analyses (part 10) when we refit under alternative priors.

In the report we will show a short code snippet like the one above, together with this explanation of the chosen options (subsampling, chains, warmup, iterations, parallelization, seed), so that it is explicit how the MCMC inference was run and why these settings are reasonable for this model and dataset.

## 7. Convergence

Fit summary

          mean    sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
    mu0   0.22  0.02    0.19     0.25       0.00     0.00   4746.92   3041.91    1.0
    mu1   9.28  9.23    1.30    21.08       0.22     1.16   3074.41   2190.62    1.0
    phi0  0.01  0.00    0.01     0.01       0.00     0.00   4881.75   3012.18    1.0
    phi1  0.09  0.03    0.04     0.16       0.00     0.00   4164.31   2974.90    1.0


The convergence diagnostics for the four NB model parameters indicate that the MCMC sampling behaved very well. All parameters have an R-hat value of exactly 1.0, which is the ideal value and suggests that the chains mixed properly and converged to a common stationary distribution. There are no signs of chain-to-chain discrepancies.

The effective sample sizes (ESS) for both bulk and tail are very high across all parameters, consistently in the range of roughly 3000–4800. Such large ESS values imply that autocorrelation in the chains is minimal and that posterior means and quantiles are estimated with low Monte Carlo error. This is also reflected in the mcse_mean and mcse_sd columns: the Monte Carlo standard errors are extremely small relative to the posterior uncertainty, indicating that numerical error from MCMC is negligible.

Parameter-wise:

- **mu0** has tight uncertainty (sd = 0.02) and extremely large ESS, showing that the background event-rate mean is estimated very reliably.
- **mu1** has substantially higher posterior variance (sd = 9.23), which is expected because drone-induced event bursts vary widely. The ESS is still large, so this variance reflects true data-driven uncertainty rather than sampling instability.
- **phi0** and **phi1** show similarly excellent convergence and very small uncertainty. These values confirm the strong overdispersion in event counts, especially for the drone class.

Overall, the diagnostics show that the MCMC run is stable, well-mixed, and free of convergence pathologies. No reparameterization or tuning adjustments appear necessary.


## 8. PPC

check ppc_nb.png

