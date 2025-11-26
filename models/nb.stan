data {
  int<lower=1> N;                       // number of observations
  array[N] int<lower=0> y;              // counts
  array[N] int<lower=0, upper=1> z;     // labels: 0 = background, 1 = drone
}

parameters {
  real log_mu0;
  real log_mu1;
  real log_phi0;
  real log_phi1;
}

transformed parameters {
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

  // Likelihood
  for (n in 1:N) {
    real mu_n  = z[n] == 1 ? mu1 : mu0;
    real phi_n = z[n] == 1 ? phi1 : phi0;
    y[n] ~ neg_binomial_2(mu_n, phi_n);
  }
}

generated quantities {
  array[N] real log_lik;

  for (n in 1:N) {
    real mu_n  = z[n] == 1 ? mu1 : mu0;
    real phi_n = z[n] == 1 ? phi1 : phi0;
    log_lik[n] = neg_binomial_2_lpmf(y[n] | mu_n, phi_n);
  }
}
