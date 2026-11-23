# msdsge

Solve, simulate and estimate **Markov-switching DSGE models** in Python.

`msdsge` is a clean-room Python implementation of the functionality of the
[RISE toolbox](https://github.com/jmaih/RISE_toolbox) (MATLAB), with model files written in a
Dynare-like syntax. It does not need MATLAB. Results are checked numerically against RISE wherever
RISE provides the feature; [docs/VALIDATION.md](docs/VALIDATION.md) lists every comparison, the
size of the difference, and the places where the two disagree.

> **Status: 0.1.0, first public version.** The first-order core (solution, filters, estimation,
> analysis) is the most thoroughly validated part. Several extensions have narrower validation or
> no RISE reference at all. Read [What is validated](#what-is-validated) before relying on a result.

## Features

| Area | What is available |
|---|---|
| Model language | Dynare-like files; Markov chains; exogenous and endogenous transition probabilities; local model variables; parameter restrictions; analytical, numerical or user-supplied steady state; `log_vars`; converter from RISE `.rs` files, including its macro language |
| Solution | First-order perturbation by functional iteration (`mfi`) or Newton (`mnk`); regime-specific or unique approximation point; anticipated shocks; orders 2 to 5; search for alternative solutions |
| Stability | Mean-square stability |
| Simulation and analysis | Regime-specific and generalised IRFs; pruned simulation; variance and historical decomposition; theoretical moments; forecasts and conditional forecasts; counterfactuals |
| Filtering | Regime-switching Kalman filter reproducing RISE, IMM and Kim-Nelson; smoothed states, shocks and regime probabilities; sigma-point filters for second-order solutions |
| Estimation | Priors by mean and standard deviation or by quantiles; posterior mode; adaptive Metropolis-Hastings, parallel tempering, DSMH and SMC; ten marginal-likelihood estimators; convergence diagnostics |
| Policy | Commitment, discretion, loose commitment, optimal simple rules |
| Occasionally binding constraints | Piecewise-linear solution through regimes |
| VARs | Reduced-form and structural VARs with Markov switching; DSGE-VAR |
| Tools | HDMR and Sobol indices, Monte Carlo filtering, stability maps, reporting |

## Installation

Requires Python 3.11 or newer.

```bash
git clone https://github.com/kleiton-stack/msdsge_python_solver.git
cd msdsge_python_solver
python -m venv .venv
.venv/bin/pip install -e ".[dev,fast,plot]"
.venv/bin/pytest -q
```

`fast` installs numba (compiled filter kernel); `plot` installs matplotlib (figures in reports).
Both are optional.

## Quick start

A three-equation New Keynesian model whose Taylor-rule coefficient switches between an active and
a passive regime, with an endogenous probability of returning to the active regime
(`examples/nk_ms.mod`):

```
var y pi i;
varexo e_m;
parameters beta kappa rho;

markov_chain pol(regimes=2);
parameters(pol) phi_pi;

beta = 0.99; kappa = 0.1; rho = 0.8;
phi_pi(pol,1) = 1.5;
phi_pi(pol,2) = 0.9;

transition_probabilities(pol);
  p(1,2) = 0.05;
  p(2,1) = 1/(1+exp(2-2*pi(-1)));
end;

model;
  y = y(+1) - (i - pi(+1));
  pi = beta*pi(+1) + kappa*y;
  i = rho*i(-1) + (1-rho)*phi_pi*pi + e_m;
end;

shocks;
  var e_m; stderr 0.25;
end;
```

```python
from msdsge import load

m = load("examples/nk_ms.mod")
s = m.solve()                         # solver="mnk", order=2, shock_horizon=k, steady_state="unique"

s.is_stable, s.mss_radius             # mean-square stability
m.solve_alternatives(starts=100)      # other solutions, if any

s.irf(periods=20)                     # by shock and regime
s.girf("e_m", draws=1000)             # regimes drawn along the horizon
s.simulate(periods=500, seed=0)
s.variance_decomposition()
```

Filtering and estimation need `varobs` and an `estimated_params` block in the model file and a
pandas `DataFrame` with one column per observable:

```python
f = s.filter(data)
f.loglik, f.smoothed_probs, f.smoothed_states, f.smoothed_shocks
s.historical_decomposition(f)
s.forecast(f, steps=12)

r = m.estimate(data)                  # posterior mode (maximum likelihood without priors)
r = r.sample("mh", draws=20000, chains=4)
r.summary(); r.rhat(); r.log_mdd("bridge")
```

Converting a RISE model:

```python
from msdsge.convert import rs_to_mod
text = rs_to_mod("model.rs", flags={"some_macro_flag": True})
```

### Model-file blocks

`var`, `varexo`, `varobs`, `parameters`, `markov_chain name(regimes=N)`, `parameters(chain)`,
`transition_probabilities(chain)`, `model` (with `# name = expr;` local variables and
`steady_state(x)`), `steady_state_model`, `initval`, `shocks`, `estimated_params`,
`parameter_restrictions`, `planner_objective`, `occbin_constraints`.

Priors in `estimated_params` follow the Dynare field order
(`name, initial, lower, upper, prior_pdf, mean, std`). Writing the prior as `beta_pdf(0.90)` switches
to the RISE convention: the two numbers are the quantiles that enclose 90% of the mass.

## What is validated

RISE is distributed as obfuscated p-code, so this package was written from the published papers
and the documented interfaces, and then compared number by number with RISE (commit `1b2edfa`,
MATLAB R2024a). The reference outputs are stored in `validation/ref/` and read by the test suite;
the tests do not call MATLAB.

**Matches RISE**

| Item | Largest difference |
|---|---|
| First-order solution, several models including Smets-Wouters with two regimes | 1e-8 or better |
| Regime-specific steady states; endogenous transition probabilities | 1e-6 |
| Likelihood, filtered and smoothed regime probabilities, smoothed states and shocks | 1e-8 to 1e-15 |
| Posterior kernel with quantile priors; posterior mode | 1e-6; 4e-8 |
| Anticipated shocks; variance decomposition; conditional forecasts | 1e-8 to 1e-10 |
| Perturbation of orders 2 and 3; pruned simulation | 2e-7 relative; 4e-5 |
| Optimal policy on the RISE examples | 3e-9 to 2e-6 |
| Occasionally binding constraints, one constraint (also against Dynare OccBin) | 1e-9 |
| Reduced-form and switching structural VARs | 1e-7 to 1e-10 |
| RISE example models: 66 of 67 compared cases | 2e-6 relative |

**Checked by other means, because RISE gives no usable reference**

- Orders 4 and 5: RISE disagrees with the analytical fourth derivative in the test model; the
  package matches it and was also checked through equation residuals.
- DSGE-VAR, the divided-difference filter and counterfactuals raise errors in the local RISE.
- Marginal-likelihood estimators and the samplers beyond Metropolis-Hastings were tested on
  distributions with a known answer.

**Known differences and gaps**

- Uncertainty constant at a unique approximation point (up to 0.01).
- Second order with endogenous probabilities is implemented but RISE's result is not reproduced.
- Three of 24 IRFs in the two-constraint occasionally-binding example.
- Hybrid expectations and balanced-growth-path solutions are not implemented.

RISE's own behaviour is reproduced by default in a few places where it looks like a shortcut rather
than a modelling choice (for example, how its filter evaluates endogenous transition matrices).
These are documented in [docs/VALIDATION.md](docs/VALIDATION.md), which is the reference for
anything not listed here.

## Repository layout

```
src/msdsge/        package
tests/             test suite (pytest; slow tests: pytest -m slow)
examples/          small model files
validation/        scripts that ran RISE (validation/matlab) and the stored references (validation/ref)
docs/              VALIDATION.md (comparisons and deviations), LITERATURE.md (algorithms and sources)
references.bib     bibliography
```

To regenerate the references you need MATLAB and RISE; point `RISE_TOOLBOX` to the toolbox folder
(default `~/RISE_toolbox`).

## Versioning

The project follows [semantic versioning](https://semver.org). Until 1.0.0 the public API may
change between minor versions. Changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## Acknowledgements and references

The methods are due to their authors; this package only implements them. The main sources are
Maih (2015) for the perturbation solution of regime-switching models, Farmer, Waggoner and Zha
(2011) and Costa, Fragoso and Marques (2005) for solutions and mean-square stability, Kim (1994)
and Hashimzade, Kirsanov, Kirsanova and Maih (2024) for filtering, and Debortoli, Maih and Nunes
(2014) for loose commitment. Full references with DOIs are in
[docs/LITERATURE.md](docs/LITERATURE.md) and `references.bib`.

RISE (Rationality In Switching Environments) is the work of Junior Maih. The example models under
`validation/` were converted from the RISE distribution and the files in `validation/ref/` are
outputs of RISE; see [NOTICE](NOTICE).
