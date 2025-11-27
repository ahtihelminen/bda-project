## 3. Model (Hierarchical NB)

The hierarchical model extends the simple two-class Negative Binomial classifier by allowing each spatial patch to have its own event-rate offset. This captures the fact that different regions of the sensor may fire more or less frequently even when no drone is present. The observation model is:

$
y_n \mid z_n = c,\; \text{patch}=i \sim \text{NegBinom}(\mu_{c,i},\, \phi_c)
$

with class-specific dispersion parameters $\phi_0, \phi_1$, and patch-specific means:

$
\log \mu_{c,i} = \log \mu_c + b_i,
$

where each patch deviation $b_i$ is shared by both classes. The hierarchical structure is completed with:

$
b_i \sim \mathcal{N}(0,\, \sigma_b), \qquad
\sigma_b \sim \text{half-Normal}(0,1).
$

This allows the model to learn how “active” or “quiet” each patch typically is, improving prediction especially when background activity varies significantly across the sensor.


## 4. Priors (Hierarchical NB)

We use the same weakly informative priors for the global parameters as in the simple NB model:

$
\log \mu_c \sim \mathcal{N}(0,\, 2^2), \qquad
\log \phi_c \sim \mathcal{N}(0,\, 1^2).
$

These priors regularize the global mean and dispersion parameters while still allowing the data to freely determine the relative scale between the background and drone classes.

For the hierarchical components we add:

$
b_i \sim \mathcal{N}(0, \sigma_b),
\qquad
\sigma_b \sim \text{Normal}^+(0,1).
$

The patch-level offsets $b_i$:

- encode the idea that patches differ in baseline activity,
- are shrunk toward zero when evidence is weak, preventing overfitting,
- share information across all patches through $\sigma_b$, allowing data-rich patches to influence data-poor ones.

The half-Normal prior on $\sigma_b$:

- is proper and weakly informative,
- favors small to moderate variability across patches,
- but does not forbid larger deviations if strongly supported by the data.

Together, these priors ensure the model remains stable while capturing meaningful spatial variation.


## 5. Stan Code (Hierarchical NB)
check models/hierarchical_nb.stan

## 6. MCMC Inference (Hierarchical NB Model)

For the hierarchical NB model we fit a more expressive observation model:

- Each patch $i$ has its own log-rate offset $b_i$.
- These offsets follow a Gaussian prior $b_i \sim \text{Normal}(0,\sigma_b)$.
- The global means satisfy $\log \mu_{c,i} = \log \mu_c + b_i$ for classes $c \in \{0,1\}$.
- A weakly informative half-Normal prior $\sigma_b \sim \text{Normal}^+(0,1)$ regularizes the patch-level effects.

        model = CmdStanModel(stan_file=hierarchical_nb.stan)
        fit = model.sample(
            data={
                "N": len(y),
                "K": n_patches,
                "y": y,
                "z": z,
                "patch_idx": patch_idx + 1,
            },
            chains=4,
            parallel_chains=4,
            iter_warmup=1000,
            iter_sampling=1000,
            seed=1337,
        )

The inference procedure is analogous to the simple NB model but adds patch-indexed parameters. We again use `cmdstanpy` with the following configuration:

- **Subsampling:**  
  We limit the dataset to `max_samples = 30000` rows.  
  This is necessary because the hierarchical model introduces hundreds of latent patch offsets $b_i$, making MCMC substantially more expensive.  
  The random subsample still captures the distribution of event counts across diverse patches.

- **Chains and draws:**  
  - 4 parallel chains  
  - 1000 warmup iterations  
  - 1000 sampling iterations  
  This results in 4000 draws per parameter, providing adequate precision for both global parameters ($\log \mu_c$, $\log \phi_c$, $\sigma_b$) and the patch-level offsets $b_i$.

- **Parallelization:**  
  `parallel_chains = min(chains, 4)` uses up to 4 CPU cores for chain execution.

- **Patch indexing:**  
  The fitting function remaps patch IDs to consecutive integers $1,\ldots,K$, so Stan only allocates offsets for patches actually present in the subsample.  
  This reduces unnecessary parameters and improves performance.

- **Saved outputs:**  
  As with the simple NB model, the Stan fit is converted to an ArviZ `InferenceData` object and saved as `hierarchical_nb_idata.nc`.  
  This file includes:
  - all global parameters  
  - all per-patch offsets $b_i$  
  - generated log-likelihood values for LOO-CV  
  - the full posterior for later diagnostics and PPC.

This setup provides a consistent Bayesian workflow while keeping the hierarchical model computationally feasible.


## 7. Convergence

Fit summary for the hierarchical NB model:

            mean    sd    hdi_3%  hdi_97% mcse_mean  mcse_sd   ess_bulk   ess_tail   r_hat
    mu0      0.02   0.00    0.01     0.02      0.00     0.00     420.19     946.76    1.01
    mu1      1.28   2.61    0.04     3.91      0.05     0.38    1450.99    2463.84    1.00
    phi0     0.02   0.00    0.02     0.03      0.00     0.00     452.94    1015.77    1.01
    phi1     0.59   0.96    0.03     1.82      0.02     0.08    1177.85    2173.97    1.00

The hierarchical model converged satisfactorily:

- **R-hat values:**  
  All parameters have $ \hat{R} \in [1.00, 1.01] $, indicating good chain mixing and no evidence of non-convergence.

- **ESS (bulk & tail):**  
  Although some global parameters (e.g., $\mu_0$ and $\phi_0$) have lower ESS than the simple NB model, they are still comfortably above the common rule-of-thumb threshold (ESS > 200).  
  The higher-variance parameters ($\mu_1$, $\phi_1$) have very large ESS (1000–2500), ensuring reliable quantile estimates.

- **Posterior uncertainty:**  
  The larger posterior variance in $\mu_1$ and especially in $\phi_1$ is expected:  
  - the drone class exhibits heavy-tailed count behavior,  
  - and the hierarchical model allows patch-specific deviations.  
  These properties are reflected in wider HDIs without indicating poor convergence.

- **mcse_mean and mcse_sd:**  
  Monte Carlo standard errors remain small relative to posterior uncertainty across all parameters.  
  This means the posterior summaries are stable and accurately estimated despite the higher model complexity.

Overall, the hierarchical NB model shows **clean convergence**, with some natural reduction in ESS due to the additional patch-level parameters. There is no indication of pathological behavior, divergences, or poor chain mixing, and the posterior estimates can be considered reliable for downstream tasks such as PPC, model comparison, and predictive evaluation.


## 8. PCC

