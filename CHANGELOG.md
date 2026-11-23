# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[semantic versioning](https://semver.org).

## [0.1.0] - 2026-11-23

First public version.

### Added
- Dynare-like model language with Markov chains, exogenous and endogenous transition
  probabilities, parameter restrictions and several ways to supply the steady state.
- Converter from RISE `.rs` model files, including the macro language.
- First-order perturbation solution (functional iteration and Newton), mean-square stability,
  anticipated shocks, unique approximation point, search for alternative solutions.
- Perturbation of orders 2 to 5 with pruned simulation.
- Regime-switching filters (RISE-compatible, IMM, Kim-Nelson), smoothing, sigma-point filters.
- Bayesian estimation: priors, posterior mode, samplers, marginal-likelihood estimators,
  diagnostics.
- Impulse responses, generalised impulse responses, variance and historical decomposition,
  theoretical moments, forecasts, conditional forecasts, counterfactuals.
- Optimal policy (commitment, discretion, loose commitment, optimal simple rules).
- Occasionally binding constraints through regimes.
- Reduced-form and structural VARs with Markov switching; DSGE-VAR.
- Sensitivity tools (HDMR, Sobol indices, Monte Carlo filtering), reporting.
- Validation against RISE with stored reference outputs; see `docs/VALIDATION.md`.

[0.1.0]: https://github.com/kleiton-stack/msdsge_python_solver/releases/tag/v0.1.0
