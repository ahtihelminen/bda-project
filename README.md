# Introduction

Event cameras output asynchronous brightness-change events rather than conventional frames, making them well suited for detecting small, fast targets such as drones. Their high temporal resolution (µs scale) allows aggregating events in short windows (e.g., 5–10 ms) and counting how many occurred within each spatial patch of the sensor.

Drone motion produces a distinctive burst of events relative to the slowly varying background. This motivates a Bayesian formulation: treat the observed event count in each patch/time window as evidence for or against the presence of a drone.

---

# MVP Bayesian Model for Drone vs Background

## Data

Divide the sensor into an $M\times N$ grid. For each patch $i$ and time window $t$:

- $y_{i,t}$: number of events in the window  
- $z_{i,t} \in \{0,1\}$: label (0 = background, 1 = drone present)

The training table consists of tuples $(y_n, z_n)$.

---

## Generative model

We assume that event counts follow different distributions when the drone is absent/present. Let class index $c \in \{0,1\}$.

**Likelihood**

$
y_n \mid z_n=c \sim \text{NegBinom}(\mu_c,\;\phi_c)
$

where:
- $\mu_c$ = mean event count for class $c$  
- $\phi_c$ = dispersion (NB parameterisation using mean + shape)

This allows overdispersion compared to Poisson.

---

## Priors

Use weakly informative priors on log-scale:

$
\log \mu_c \sim \mathcal{N}(0,\,2^2)
$

$
\log \phi_c \sim \mathcal{N}(0,\,1^2)
$

Both classes get their own means and dispersions. All priors are proper and simple enough for the MVP.

---

## Posterior

Given observed training data $D=\{(y_n,z_n)\}$, sample from the joint posterior:

$p(\mu_0,\mu_1,\phi_0,\phi_1 \mid D)$

For prediction, for a new observation $y^*$:

$p(z^*=1 \mid y^*, D)=\frac{ p(y^* \mid z^*=1, D)\; p(z^*=1)}{
\sum_{c\in\{0,1\}} p(y^* \mid z^*=c, D)\; p(z^*=c)
}
$

where:

- $p(z^*=1)$ = empirical class frequency or a prior class probability
- $p(y^* \mid z^*=c, D)$ = posterior predictive NB density

Posterior predictive is computed by averaging over posterior samples:

$p(y^* \mid z^*=c, D)=\int \text{NegBinom}(y^* \mid \mu_c, \phi_c)\;
p(\mu_c,\phi_c \mid D)\; d\mu_c d\phi_c.$

This quantity is the basic detection score for each patch and time window.


# Get Started
Start by installing python package and project manager [UV](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.
Next clone the repo using

    git clone https://github.com/ahtihelminen/bda-project

Navigate to repo root

    cd bda-project

Setup the environment using UV

    uv sync
This should automatically create a .venv with dependencies specified in `pyproject.toml`.
Next download the fred-0 dataset named 0.zip from this [Drive](https://drive.google.com/drive/folders/1EzP4ATeHGT5gtHDnABxO0fWyyLeyuxtM?usp=sharing). It's otherwise the same as the one with same name in the official FRED Drive but it has a .dat conversion of the event data which we need for this project.
Once the download is complete and you have extracted the zip we need to generate labels that are fitting for our usecase. Do this by running

    uv run scripts/fred_labels.py \
    --labels-txt ./0/interpolated_coordinates.txt \
    --dat ./0/Event/events.dat \
    --window-ms 50 \
    --grid-rows 180 \
    --grid-cols 320 \
    --out ./data/labels_0_50_180_320.csv

The command expects to find the 0 data folder in the repo root, modify this to the actual location if needed. The output should be the labels csv under a `data` folder and this text to terminal

    time_bin range in final label table: 270 → 2221 (total_bins=2232)
    Saved label CSV → ./data/labels_0_50_180_320.csv (rows=350881)

Next we do the actual model fitting using

    uv run scripts/fit_mvp.py \
    --window 50 \
    --grid-rows 180 \
    --grid-cols 320 \
    --labels-csv ./data/labels_0_50_180_320.csv \
    --dat ../fred/0/Event/events.dat

NOTE: This takes a while (~1-2 min), be patient...
Finally, the output should look like this

    Exported MVP table with 128563200 rows to ./data/counts_0_50_180_320.parquet
    Using subset of 10000 rows out of 128563200 for MAP.
    MAP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% 0:00:03 logp = -1,304.3, ||grad|| = 0.00016224
    Fitted MVP Negative Binomial model parameters (MAP):
    mu0: 0.1184
    mu1: 6.5907
    phi0: 0.0055
    phi1: 0.0828
    log_mu0: -2.1341
    log_mu1: 1.8857
    log_phi0: -5.1948
    log_phi1: -2.4917

This fitted the model parameters and wrote parquet file containing the event counts for each time and space bin. Next time we can run 

    uv run scripts/fit_mvp.py \
    --window 50 \
    --grid-rows 180 \
    --grid-cols 320 \
    --labels-csv ./data/labels_0_50_180_320.csv \
    --counts-parquet ./data/counts_0_50_180_320.parquet

to make the fitting faster.
Finally we test the model by running

    uv run scripts/main.py ../fred/0/Event/events.dat \
    --window 50 \
    --grid-rows 180 \
    --grid-cols 320 \
    --mu0 0.1184 \
    --mu1 6.5907 \
    --phi0 0.0055 \
    --phi1 0.0828 \
    --prior-drone 0.003

This should start playing the event footage and drawing a probability heatmap to areas where it thinks a drone exists.
