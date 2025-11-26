data {
  int<lower=1> N;                 // number of observations
  int<lower=1> K;                 // number of patches used in subsample
  array[N] int<lower=0> y;        // counts
  array[N] int<lower=0, upper=1> z;     // labels: 0 = background, 1 = drone
  array[N] int<lower=1, upper=K> patch_idx; // patch index for each obs (1..K)
}

parameters {
  // Global class-specific means (log-scale)
  real log_mu0;
  real log_mu1;

  // Dispersion parameters (log-scale)
  real log_phi0;
  real log_phi1;

  // Non-centred patch-level random effects
  real<lower=0> sigma_b;
  vector[K] b_raw;
}

transformed parameters {
  vector[K] b = b_raw * sigma_b;

  // Deterministic transforms for global means and dispersions
  real<lower=0> mu0    = exp(log_mu0);
  real<lower=0> mu1    = exp(log_mu1);
  real<lower=0> alpha0 = exp(log_phi0);   // PyMC alpha0
  real<lower=0> alpha1 = exp(log_phi1);   // PyMC alpha1

  // Stan's neg_binomial_2 uses phi with Var = mu + mu^2 / phi
  // PyMC uses alpha with Var = mu + alpha * mu^2
  // => phi = 1 / alpha
  real<lower=0> phi0 = inv(alpha0);
  real<lower=0> phi1 = inv(alpha1);
}

model {
  // Priors
  log_mu0  ~ normal(0, 2);
  log_mu1  ~ normal(0, 2);
  log_phi0 ~ normal(0, 1);
  log_phi1 ~ normal(0, 1);

  sigma_b  ~ normal(0, 1);   // half-Normal via lower bound
  b_raw    ~ normal(0, 1);

  // Likelihood
  for (n in 1:N) {
    real log_mu_n;
    real mu_n;
    real phi_n;

    if (z[n] == 1) {
      log_mu_n = log_mu1 + b[patch_idx[n]];
      phi_n    = phi1;
    } else {
      log_mu_n = log_mu0 + b[patch_idx[n]];
      phi_n    = phi0;
    }

    mu_n = exp(log_mu_n);

    y[n] ~ neg_binomial_2(mu_n, phi_n);
  }
}

generated quantities {
  array[N] real log_lik;

  for (n in 1:N) {
    real log_mu_n;
    real mu_n;
    real phi_n;

    if (z[n] == 1) {
      log_mu_n = log_mu1 + b[patch_idx[n]];
      phi_n    = phi1;
    } else {
      log_mu_n = log_mu0 + b[patch_idx[n]];
      phi_n    = phi0;
    }

    mu_n = exp(log_mu_n);

    log_lik[n] = neg_binomial_2_lpmf(y[n] | mu_n, phi_n);
  }
}
