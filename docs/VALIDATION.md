# Validation and design decisions

Record of the design choices of `msdsge`, a Python package that solves and estimates DSGE models
with Markov regime switching (MS-DSGE) with the functionality of the RISE toolbox, and of the
numerical comparison of each module against RISE. The entries were written between 2026-11-22 and
2026-11-23 and are organised here by topic. Where a later result superseded an earlier one, the
final status is given and the earlier one is mentioned.

Literature notes are in `docs/LITERATURE.md`; the bibliography is in `references.bib`.

Contents

1. [Scope and principles](#1-scope-and-principles)
2. [Roadmap](#2-roadmap)
3. [Solver conventions and first-order solution](#3-solver-conventions-and-first-order-solution)
4. [Model language, converter and coverage of the RISE examples](#4-model-language-converter-and-coverage-of-the-rise-examples)
5. [Linear filters, likelihood and smoothing](#5-linear-filters-likelihood-and-smoothing)
6. [Perturbation of order 2 to 5](#6-perturbation-of-order-2-to-5)
7. [Sigma-point filters](#7-sigma-point-filters)
8. [Analysis tools: moments, decompositions, forecasting, sensitivity, time series](#8-analysis-tools-moments-decompositions-forecasting-sensitivity-time-series)
9. [Bayesian estimation](#9-bayesian-estimation)
10. [Occasionally binding constraints](#10-occasionally-binding-constraints)
11. [Optimal policy](#11-optimal-policy)
12. [VAR models and DSGE-VAR](#12-var-models-and-dsge-var)

---

## 1. Scope and principles

1. **Goal.** Functional parity with RISE (solution, simulation, filters, estimation, forecasting,
   optimal policy, VARs, reports), without depending on MATLAB in the final product.
2. **Reimplementation, not translation.** RISE is distributed as p-code. The port is based on
   papers (see `docs/LITERATURE.md`), the help texts of the `.m` files, the readable examples
   (`examples/`) and numerical comparison of results. The p-code is not decompiled.
3. **Validation oracle.** A local RISE (commit `1b2edfa`, 2019-09-19) on MATLAB R2024a, used only
   during development to generate reference results saved in `validation/`. The package tests read
   those files; they do not call MATLAB. Scripts are in `validation/matlab/`, results in
   `validation/ref/`, tests in `tests/test_vs_rise.py` and the module-specific test files named
   below. The MATLAB scripts locate RISE through the environment variable `RISE_TOOLBOX`
   (default `~/RISE_toolbox`).
4. **Acceptance criterion for each module.** Reproduce RISE within a declared numerical tolerance
   (solution, IRFs, likelihood at the same parameter vector). Parity is numerical, not bit for bit;
   optimisers and samplers are equivalent, not identical.
5. **Model syntax.** A text file in the Dynare style (`var`, `varexo`, `parameters`,
   `model; ... end;`), with additional blocks for Markov chains (`markov_chain`,
   `parameters(chain)`, `transition_probabilities(chain)`). This is the only deliberate change
   relative to RISE. A `.rs` → `.mod` converter is provided for migration (section 4).
6. **Package name:** `msdsge` (provisional). It does not use "RISE" in the name.
7. **Stack:** Python ≥ 3.11, NumPy, SciPy, SymPy (symbolic derivatives), pandas. Acceleration
   (numba/JAX) only after the reference version is validated.

Deviations from this plan: none recorded.

## 2. Roadmap

Order of construction.

| Phase | Content | Validation |
|---|---|---|
| 0 | Structure, literature, inventory of RISE | n/a |
| 1 | Dynare-like parser, symbolic model, derivatives | example files; count of equations × variables |
| 2 | Steady state (analytical and numerical, per regime and ergodic) | RISE examples |
| 3 | First-order solution: single regime; exogenous switching; endogenous; anticipated shocks | Dynare/gensys in the constant case; RISE on the FWZ and FRWZ examples |
| 4 | Mean square stability, IRFs (per regime and generalised), simulation, moments | RISE |
| 5 | Filters (Kim-Nelson, IMM, switching Kalman), likelihood, smoothing | RISE at the same parameter vector |
| 6 | Priors, mode, MCMC, marginal density, diagnostics | comparable posterior |
| 7 | Perturbation of orders 2 to 5, pruning, nonlinear filters | third-order examples of RISE |
| 8 | Forecasting and conditional forecasting, historical and variance decompositions | RISE |
| 9 | Optimal policy (commitment, discretion, loose commitment, simple rules), OccBin via regimes | RISE examples |
| 10 | DSGE-VAR, RFVAR/SVAR, HDMR, Monte Carlo filtering, time series and reports | RISE examples |

## 3. Solver conventions and first-order solution

### 3.1 Conventions (after the literature review and the inventory)

1. **Solver conventions** (Maih, 2015): `Q[i,j] = Pr(s_{t+1}=j | s_t=i)`; regime-specific steady
   state as the approximation point; "backward" initial guess; `mfi` as the default solver, `mnk`
   (Newton with Kronecker) as the alternative. `mn` (TFQMR), `fwz`, Schur/QZ and Gröbner are left
   for later.
2. **Endogenous probabilities at first order:** they enter evaluated at the steady state of the
   origin regime (the derivative of the probability multiplies a zero residual). Confirmed against
   RISE in validation case B; see the caveat in section 6 on models with a non-zero steady state.
3. **MSS:** computed with Q at the steady state. With endogenous probabilities it is only
   informative, because there is no established criterion (Maih, 2015, §5.3).
4. **The default filter was planned to be the IMM**, with Kim-Nelson as the alternative
   (Hashimzade et al., 2024). The filter that reproduces RISE is named `rise` (section 5).
5. **Incomplete specification.** The inventory showed that only 39% of the RISE help texts are
   useful; `solve`, `estimate`, `filter`, `forecast`, `irf`, `set` and the list of options and
   defaults are known only from their use in the examples or by running RISE. The list of options
   is extracted from running MATLAB (`method(rise.empty(0))`) and saved in `validation/`: the RISE
   options and defaults are in `validation/ref/rise_defaults.tsv` (150 lines).
6. **Validation done without MATLAB:** the number of stable solutions of FWZ (2011) equals 1, 2
   and 4 in the three parametrisations, as in Table 2 of Maih (2015).

### 3.2 Results against RISE

| Item | Result |
|---|---|
| First order, FWZ (3 parametrisations), `mnk` and `mfi` | equal (1e-8; the RISE `mfi` stops at 1.5e-8, `fix_point_TolFun`) |
| Regime-specific steady state (case A) | solution, steady state and likelihood equal (1e-11) |
| Endogenous probabilities (case B) | solution equal; the hypothesis of item 2 above confirmed |
| Anticipated shocks (`solve_shock_horizon=3`, 12-variable two-regime model) | equal (1e-10) |
| IRFs per regime | equal |

**Deviation from RISE: `mfi` tolerance.** 1e-12 here against 1.5e-8 in RISE. The stopping
criterion of `mfi` is relative to the size of T (with an absolute value, EDO, whose coefficients
are of the order of 1e3, did not converge).

**Solver changes motivated by a larger model** (two regimes, one with a pegged interest rate):
minimum-norm solution when U_r is singular at the initial guess, Newton on the state columns with
line search and warm start from functional iterations; `solve_alternatives` to enumerate
solutions.

### 3.3 Unique approximation point

`solve(steady_state="unique")`: steady state with the parameters at the ergodic mean, plus the
uncertainty constant (Maih, 2015, eq. 13). `steady_state=None` follows the `unique` attribute of
the `steady_state_model` block.

**Open: uncertainty constant at the unique point.** In the RBC of Foerster et al. (2016), the
steady state, T and R match RISE (1e-6), and so does the constant of the exogenous variable. That
of the forward-looking variables differs by up to 0.01 (residual of 4e-4 in the Euler equation
with the RISE constant). The origin was not identified.

With parameters computed inside the steady-state block, the unique point uses those of the ergodic
point in all regimes; there is no RISE example to check this.

### 3.4 Items listed as open for the first-order solver

- A version that exploits static variables and λ_x (the unknown is the full T, n x n).
- Uncertainty term T_σ, future-regime parameters, ergodic or imposed approximation point (the
  unique point and the imposed steady state were added later; see 3.3 and section 4).
- Shock leads and lags (added later through auxiliary variables; see section 4). Anticipated shocks
  were added and validated (table above).
- Numerical comparison of T and R with Dynare in the single-regime case.

## 4. Model language, converter and coverage of the RISE examples

### 4.1 What is implemented

- **Converter from `.rs` to the package syntax** (`msdsge.convert.rs_to_mod`), with macros, local
  definitions and priors by quantiles. Whatever is not converted raises an explicit error.
- `parser.py`, `model.py`: block `parameter_restrictions; expr >= expr; end;` (inequalities, with
  `name(chain,k)`), stored in `ModelSpec.restrictions` and evaluated by `Model.check_restrictions`;
  `steady_state_model(imposed, unique, loop)` with parameters computed in the block and auxiliary
  constants (`# name = expr;`, which is where the `xx_ssmdef_N` go); `model(linear)` (only
  recorded: it does not change the solution); `log_vars`; an observable that is exogenous; shocks
  with leads and lags (auxiliary variable `AUX_EXO_name`); a parameter that is declared and not
  used may be left without a value, as in RISE.
- Steady-state file: a function `f(model, {parameter: value}) -> {name: value}` (variables and new
  parameters; attributes `f.imposed`, `f.unique`), passed as `solve(steady_state_file=f)`; also
  `solve(steady_state_imposed=True)`. Hand ports of `fs2000_steadystate.m` and
  `steady_state_4_Canonical_Const.m` are in `validation/examples_ssfiles.py`.
- `solve(log_vars=[...])` (`solve_log_approx_vars`): T and R rescaled by the level of the
  approximation point. Here the solution then describes log(x) (steady state, simulation and
  filter in logs); RISE returns the same IRF, but stores the steady state and simulates in levels.
- Filter: an observed exogenous variable enters as data (time-varying intercept, no variance);
  measurement error through the parameter `stderr_<observable>` when `measurement_errors` is not
  given. The numba path is switched off when there is an observed exogenous variable.
- Estimation: the prior equals -inf when some parameter restriction is violated.
- Converter: the blocks above, the argument `parameters=` (equivalent to `set(m,'parameters',p)`,
  accepts `name_chain_state`) and chains declared in alphabetical order, which is how RISE numbers
  composite regimes (without this the regimes of models with two chains came out swapped).
- `validation/examples_config.py` (flags, parameters and options of each driver),
  `validation/gen_ref_examples.py` (runs RISE on every case and writes `ref/examples.mat`) and
  `validation/sweep_examples.py` (converts, solves and compares steady state and IRFs with RISE).

### 4.2 Coverage

A first sweep of the 88 `.rs` files in the RISE `examples/` folder gave: 14 solved directly,
including Smets-Wouters with 40 variables and 2 regimes; 43 depended on parameters, flags or files
defined in the MATLAB scripts of the example; 27 used blocks not yet converted
(`planner_objective`, `parameter_restrictions`, parameters computed in `steady_state_model`,
attributes `imposed`/`unique`, `log_vars`, OccBin); 4 needed a supplied steady state (a `.m` file
in RISE) or the unique point.

Current coverage (`validation/ref/sweep_examples.txt`): 89 model files (88 `.rs` and 1 `.rz`), 99
cases counting flag variants. 80 cases solve (77 stable and 3 with MSS radius >= 1), in 70 files;
before there were 14. Of the remaining 19 files: 16 are `@#include` fragments; `rbcns.rs` is not
calibrated anywhere; `emosp_ns.rs` is non-stationary (the local RISE does not find the steady
state either); `sgusea1.rs` depends on a steady-state `.m` file that was not ported, and the local
RISE fails to read the model (macro error), so there is nothing to tabulate.

### 4.3 Validation against RISE

Steady state by name and IRFs per regime, unit shock, 10 periods: 67 cases compared; in 66 the
largest IRF difference relative to scale is below 2e-6 (65 below 1e-6; the worst are the `emosp`
family, 1.5e-6, and the unique point of FRWZ, 2.8e-7) and the steady-state difference is below
2e-6. They include Smets-Wouters with 2 regimes (1.3e-13), Tao Zha's models with
`parameter_restrictions` (up to 1.4e-7, the tolerance of the RISE `mfi`), `globdynzlb`,
`nkp_switch` (4 regimes), LWZ, EDO (97 variables) and `log_vars`. Tests:
`tests/test_examples_vs_rise.py` (12 cases) and `tests/test_language.py`.

### 4.4 Does not match or was not validated

- `hybrid_expect.rs`: difference of 0.18 in the IRFs. Hybrid expectations (`hbe_lambda`, `hbe_w`)
  were not implemented; the RISE README marks them as obsolete.
- `woodford2003` with `multiple=true` and `stata/model11`: RISE returns code 21 (iterations
  exhausted or multiple solutions); the package returns a stable solution. Not investigated.
- The local RISE (commit 1b2edfa) does not read `steady_state_model(imposed)`, `model(linear)` or
  `$(x)` (`constant.rs`, `ModelWithMeasurementErrors/Canonical_Const.rs`, `tshocksnk.rs`): these
  three solve here, without a reference. For that reason the attributes `imposed`, `unique` and
  `loop` of the block have only unit tests; the option `steady_state_imposed` was validated with
  `globdynzlb`.
- Observed exogenous variables and measurement errors in the filter: internal consistency tests
  only (trend removed by hand; explicit `measurement_errors`). The likelihood of `ireland2004` was
  not compared with that of RISE; the treatment of the first period is an assumption of this
  package.
- Models with a `.m` file that was not ported (`emosp`, `targets`, `pdk`, `sgusea12`, `edo`,
  OccBin): the sweep uses the steady state and the new parameters returned by RISE, tabulated in
  `examples.mat`. This holds only at the driver's parameters; to estimate, the file must be ported.
- The OccBin cases in the sweep solve only the reference regime.

**Left out:** hybrid expectations, balanced growth path (`solve_bgp`), `log_vars` above first
order, equalities in `parameter_restrictions`, `level_variables` and `exogenous_definition` (no
syntax example).

## 5. Linear filters, likelihood and smoothing

### 5.1 Results against RISE

| Item | Result |
|---|---|
| Filter, single regime | increments equal (1e-15) |
| Filter with switching | increments equal (1e-11) with `kf_riccati_tol=0` in RISE |
| Updated and smoothed probabilities | equal |
| Smoothed states and shocks (latent state, with switching) | equal (1e-10). RISE smooths each regime separately and weights by the smoothed probabilities; shock of the first period = 0 |
| Riccati shortcut (`riccati_tol`) | increments equal (4e-15) at the 49 tolerances from 1e-12 to 1 and at the default reference (`filter.mat`, field `ms`) |
| `presample`, `init_variance`, `user_init` with means only | equal (1e-13), including the two unit-root models under `kf_init_variance` |
| Diffuse Lyapunov | equal (1e-15) to the function `diffuse_lyapunov_equation` in 4 cases |

RISE behaviours identified by numerical comparison and reproduced in the `rise` filter:

1. Initial state and covariance common to the regimes; Lyapunov covariance with T and R averaged
   over the ergodic distribution; the first observation uses the initial state as the predicted
   one.
2. Mixing across regimes without the dispersion-of-means term.
3. With endogenous probabilities, the whole matrix Q is evaluated at the updated state of the first
   regime; smoothing uses the Q of the transition arriving at t.

Items 2 and 3 look like shortcuts or defects of the 2019 version, not theoretical choices. The
`imm` and `kim` filters of the package do not reproduce them.

### 5.2 Riccati shortcut

RISE freezes gain and covariance once they converge (`kf_riccati_tol=1e-6`). The stopping rule
could not be reproduced at first, and the package ran the full recursion; the rule was identified
later and is now available as `riccati_tol`.

**Rule.** The Kalman gain of all regimes is compared with that of the previous period, with the
shock rows included (the RISE state is augmented); when the largest absolute difference falls
below the tolerance, and there are no missing observations ahead, gain, F and covariances are
frozen. With regime switching the gain does not converge, it only passes close by chance: in the
test case the freeze happens at the 24th observation and changes the likelihood by 0.026 (in 600
observations). For that reason the package default is 0.

Pending: the Riccati shortcut in the numba kernel (with `riccati_tol > 0` the filter uses the
Python loop).

### 5.3 Initialisation options

`_initial_cov` calls the diffuse Lyapunov equation when there is a unit root (before, `diffuse·I`).
`run_filter` has `riccati_tol`, `presample`, `init_variance`, `user_init`.

Does not match or could not be compared:

- **Unit root with default initialisation:** the local RISE returns retcode 23 for any
  `lyapunov_diffuse_factor`, and `lyapunov_algo='schur'` is "under revision". The package uses the
  diffuse Lyapunov equation, which could only be checked against the isolated RISE function, and
  that function only runs with the patch `symmetrize.m`. With a Jordan block at the unit root RISE
  gives up (10·I); the package applies the same formula.
- **`kf_user_init` with covariance or probabilities:** fails in RISE (function `issvector` does
  not exist). Implemented, without comparison. Variables missing from the dictionary stay at the
  default (not checked).
- **`kf_ergodic=false`:** uniform `init_probs` gives a difference of 0.037 in the likelihood
  (-404.567 against -404.530). Rule not identified.

### 5.4 Smoother of `imm` and `kim`

Hashimzade et al. (2024), Algorithm 4. Without regime switching it coincides with that of the
`rise` filter (1e-10). In simulation (8 samples of 400 observations, measurement error 0.5), the
RMSE of the latent state falls from 0.384 (updated) to 0.366 (`imm`) and 0.367 (`kim`); the `rise`
smoother gives 0.363. With measurement error 0.1 none of the three improves on the updated state
(0.095 to 0.096 against 0.094). There is no gain over `rise` in this model.

### 5.5 Performance

About 50 ms per posterior evaluation in a small model with 400 observations with the pure Python
filter loop. With the filter compiled with numba (optional): from 50 ms to 2.7 ms in the test
model with 400 observations.

## 6. Perturbation of order 2 to 5

### 6.1 What is implemented

`src/msdsge/perturbation.py`: solution of order p (up to 5) by truncated polynomials in
`w = (z, sigma*eps')`, derivatives by SymPy (non-zero entries only, one representative per
symmetry class), coefficients stored by monomial, system
`U_r X_r + sum_q Q[r,q] Ap_r X_q C_rq + A_r = 0` solved by Kronecker (up to 1500 unknowns) or
matrix-free GMRES, with TFQMR as fallback; endogenous probabilities at any order; `simulate` (with
and without pruning, with imposed shocks, regimes and initial condition), `irf`, nonlinear `girf`
and `simulated_moments`. Tests in `tests/test_perturbation.py` (44); references in
`validation/ref/pert.mat` and `pert2.mat` (`validation/matlab/ref_pert.m`, `ref_pert2.m`);
exploration in `validation/explore_perturbation.py` and `explore_pruning.py`.

`solve_first_order(order=...)` accepts 2 to 5 (exogenous order 2 remains in `order2.py`, and `G2`
is filled in both cases); the result is in `Solution.pert`, and `Solution.simulate` delegates to
`perturbation.simulate` when `pert` is present.

### 6.2 Results against RISE

| Item | Result against RISE |
|---|---|
| Second order, constant parameters (`order2.py`) | equal (1e-6), including the σσ term |
| Second order, regime-specific steady state (`order2.py`) | equal (1e-6) after including, in the σ column of the state law, the difference between the steady states of the regimes |
| Tz, Tzz, Tzzz in `o2c`, `o2` (regime-specific steady state), `small_nk` (Binning), `frwz_nk` (regime-specific steady state) | equal: relative difference below 2e-7 (the RISE `mfi` stops at 1.5e-8) |
| Second order of the new module against `order2.py` | equal (1e-15) |
| Simulation of order 1 to 3, with and without pruning, same shocks, regimes and starting point | equal (up to 3e-6 in `o2c` and `o2`, 4e-5 in `small_nk`; `small_nk` without pruning at third order only in the first 20 periods, after which it explodes in both) |
| IRFs of order 3 | compatible within noise: the RISE IRF at higher order is a Monte Carlo average (`irf_draws=50`) |

### 6.3 Does not match, and why

1. **Orders 4 and 5 of RISE are wrong** (commit 1b2edfa). In `o2c`, `A = A(-1)^rho exp(sig E)` has
   analytical fourth derivative -0.2079; the package gives -0.2079 and RISE -0.5643. The terms
   without sigma already differ. Orders 4 and 5 of the package were validated by the analytical
   derivative, by the vanishing of the coefficients of degree 2 to p of the equations and by the
   residual of the equations in levels, which falls as t^(p+1).
2. **Endogenous probabilities.** In the solution, RISE evaluates the probabilities with K (a
   purely predetermined state) at the steady-state value of A, although it reports the correct
   matrix in `solution.transition_matrices.Q`. Verified in 5 models (`o2r`, `o2e`, `o2x`, `o2y`,
   `o2w`): imposing that point, the first order of RISE is reproduced (1e-7); A and C are read
   correctly. The earlier case B did not detect this because it had a zero steady state. The
   general rule of the index swap was not identified. The endogenous second order of RISE was not
   reproduced even with that point (the terms in K vanish and those in A have a different scale).
   That of the package passes the independent residual check (model `o2r`, orders 2 and 3). **Any
   model whose probabilities depend on a variable with a non-zero steady state should be checked
   for this.**
3. **Pruning at the regime switch.** Within a regime the pruned simulation matched RISE from the
   start. The treatment of the switch in the first implementation (re-anchoring the deviations at
   the steady state of the new regime) was a choice of this package that had not been compared with
   RISE. At the switch, RISE adds the difference of steady states to all pruned components (in
   `o2`, K jumps from 27.8 to 41.7 on entering the regime with steady state 12.7; without pruning
   it goes to 26.8). The package keeps the sum only in the first-order component (default, same as
   the earlier code); `regime_switch="rise"` reproduces RISE.
4. **IRFs per regime with regime-specific steady state:** those of RISE start from a point that is
   not the steady state of the regime and carry transition dynamics; not reproduced.

### 6.4 Out of scope

Unique approximation point (`steady_state="unique"`) at order 3 to 5 and with endogenous
probabilities (`NotImplementedError`; the exogenous second order remains in `order2.py`), which
excludes the `howto.m` of FRWZ, which uses `steady_state_unique`. Probabilities that depend on
variables at t+1. Theoretical moments of higher order: RISE does not have them
(`theoretical_autocovariances` warns "Only first order will be used" and returns the same result
at orders 1 to 3); the package offers simulated moments.

### 6.5 Solution time

Seconds; package = symbolic derivatives, once per model, plus the system per order; RISE = full
`solve` per order, not counting the reading of the model with the derivatives.

| Model | derivatives | order 2 | 3 | 4 | 5 | RISE orders 1 to 5 |
|---|---|---|---|---|---|---|
| `o2c` (n=3, h=2) | 0.49 | 0.02 | 0.04 | 0.06 | 0.15 | 0.25; 0.49; 0.84; 0.94; 2.16 |
| `o2` (n=3, h=2) | 0.10 | 0.01 | 0.02 | 0.09 | 0.21 | 0.23; 0.53; 0.54; 1.56; 2.99 |
| `small_nk` (n=11, h=1) | 3.70 | 0.01 | 0.02 | 0.14 | 0.05 | 0.50; 0.45; 0.31; 0.53; 6.60 |
| `frwz_nk` (n=3, h=2, up to third) | 1.32 | 0.004 | 0.007 | | | 0.21; 0.53; 1.90 |

## 7. Sigma-point filters

Files: `src/msdsge/nonlinear_filters.py`, `src/msdsge/filter_extras.py`,
`tests/test_nonlinear_filters.py`, `validation/explore_nlf.py`, `validation/explore_riccati.py`,
`validation/gen_o2_data.py`, `validation/matlab/{ref_nlf,ref_riccati,ref_init}.m`, helpers
`nlf_probe.m`, `nlf_logff.m`, `flt_probe.m`, `symmetrize.m`, models `o2f/o2cf/o2mf.rs`,
`flt_ur.rs`, `flt_urms.rs`; references `validation/ref/{nlf,riccati,filter_init}.mat` and data
`nlf_*_data.csv`, `filter_ur_data.csv`. (The Riccati and initialisation results are in section 5.)

**Identification method.** `dsge.filter` accepts `kf_user_algo`; a user filter that only records
its inputs reveals the initial state and covariance and the solution matrices, and wrapping the
handle of the law of motion records every sigma point evaluated. No p-code was read.

| Item | Result against RISE |
|---|---|
| Filter selection at order > 1 | There is no `kf_algorithm`. The default is the `switching_divided_difference_filter`; the others enter through `kf_user_algo` (`@switching_unscented_kalman_filter`, `@switching_cubature_kalman_filter`) |
| UKF and CKF, recursion with the solution matrices of RISE (`o2c`, `o2`, `o2m`, 200 obs.) | increments equal (6e-13 to 3e-11) |
| UKF and CKF, second-order solution of the package | increments within 2e-6 (`o2c`), 4e-6 (`o2m`), 2e-4 (`o2`); the difference comes from the tolerance of the RISE solver (1.5e-8) |
| DDF | **no comparison**: the RISE p-code gives a syntax error on R2024a, and therefore the default filter of order > 1 does not run on the local RISE. Implemented from the paper; likelihood within 3e-4 of the UKF in `o2c` and `o2m` |

RISE details reproduced in the sigma-point filters: state `[y; shocks]` in the order static,
predetermined, "both", forward-looking; lower Cholesky; floor `sqrt(eps)` on the eigenvalues of
the predicted covariance and before the factorisation; initialisation of the linear filter; law of
motion without pruning. `o2` (steady states very far apart) is a degenerate case in both programs:
the probability goes to one regime and the likelihood is -845.

A sigma-point filter in `EstimationProblem` implies `order=2`.

**Pending.** Smoothed states and shocks in the sigma-point filters; sigma-point filters with
endogenous probabilities, observed exogenous variables and the `stderr_X` convention.

## 8. Analysis tools: moments, decompositions, forecasting, sensitivity, time series

Files: `src/msdsge/forecasting.py`, `sensitivity.py`, `timeseries.py`, `report.py`, `display.py`,
`analysis.py`; tests in `tests/test_forecasting.py`, `test_sensitivity.py`, `test_tools.py`;
scripts `validation/matlab/ref_tools.m`, `ref_hdmc.m`, `ref_sens.m` (models `tool_c.rs`,
`tool_same.rs`) and references `validation/ref/tools.mat`, `hdmc.mat`, `sens.mat`.

### 8.1 Results against RISE

| Item | Result against RISE |
|---|---|
| Variance decomposition per regime (12-variable two-regime model, horizons 1 to 400 and infinity) | equal (1e-8) |
| Generalised IRF | compatible with that of RISE within Monte Carlo noise; not an exact comparison |
| Simulation/forecast with imposed regimes and shocks (`simul_regime`, with and without shock uncertainty; `flt_ms`, `tool_c`, `refA`) | equal (1e-10) redoing the path with the regimes and shocks returned by RISE |
| Conditional forecast on an endogenous variable, on endogenous + shock and on a shock (`forecast_cond_endo_vars`, `forecast_cond_exo_vars`) | equal (1e-9 in the variables, 1e-8 in the shocks): it is the minimum-norm solution |
| `forecast_conditional_hypothesis` (jma, ncp, nas) | identical to each other without anticipated shocks; the option was not implemented |
| `theoretical_autocovariances` with switching | **rule identified**: Lyapunov with T and R averaged over the ergodic distribution, `ac(:,:,k) = Tbar^k V`; equal (1e-10; 12-variable two-regime model 1e-9, variables in alphabetical order). Autocorrelations equal (1e-10) |
| `vardec_ergodic=true` | same rule of the averaged matrices; equal (1e-10) |
| Historical decomposition, single regime | equal (1e-9) |
| Historical decomposition with switching | **rule identified**: contribution of shock k = smoothed history minus the average of 100 simulations with all shocks except k, regimes drawn from the chain starting from the smoothed probabilities of the first period; `init` is the residual. It is Monte Carlo: two RISE calls give different numbers. `method="rise"` computes the exact expectation; against the average of 300 RISE calls the largest difference is 0.0004 (shock `e`) and 0.008 (`u`, `init`), within Monte Carlo error |
| `print_solution`, `resid` | same text after normalising whitespace (the solver line differs) |
| `hpfilter`, `aggregate`, `ma_filter`, window | equal (1e-8 or better) |
| HDMR (Ishigami and Sobol g, order 4, 75% of the sample) | `f0` and `D` identical; indices within 0.012 (the rule RISE uses to estimate the coefficients is unknown) |
| MCF, `trinity.rs`, 500 RISE draws | the region with a solution coincides with the stability region here in at least 99% of the draws |

Against analytical values: Sobol indices of the Ishigami function (error < 0.005 with order 8) and
of the g function (first order, error < 0.005).

### 8.2 Does not match or could not be compared

- **Historical decomposition with regime switching.** That of RISE is avowedly approximate and
  attributes to a shock part of a variable that does not depend on it. That of the package
  (`method="weighted"`, the default) uses average matrices weighted by the smoothed probabilities,
  is exact without switching and always adds up to the smoothed variable. `method="rise"`
  reproduces the RISE rule (table above).
- **Theoretical autocovariances.** RISE returns a single matrix for the switching model. The
  aggregation rule was not identified at first and the package computed per regime; after the rule
  was identified, `theoretical_autocovariances` has `regime=None` (the RISE rule) as the default;
  before it was regime 1. With one regime the result is the same.
- `counterfactual` and `ts/transform` fail in the local RISE itself (R2024a, "Brace indexing is
  not supported"); `vardec_theoretical=false` returns "not yet implemented". The versions here
  have internal tests only (the counterfactual with one regime reproduces the smoothed history).
- HDMR of order 8: RISE returns indices larger than one (sum close to 8); here they converge to
  the analytical ones.
- The Sobol and Halton sequences of RISE differ from those of `scipy.stats.qmc`.

**Left out:** conditional forecast with anticipated shocks compared with RISE (implemented with
`anticipated=True`, with an internal consistency test only); interval conditions compared with
RISE; `forecast_real_time`; MCF plots (`cdf_plot`, `scatter`); Chow-Lin, X-13 and the other `ts`
methods; conditional forecasting with endogenous probabilities uses Q at the steady state.

`Solution` has four shortcut methods (`theoretical_autocorrelations`, `density_forecast`,
`counterfactual`, `print_solution`); `analysis.py` has `ergodic_matrices`. matplotlib 3.11.2 is
used only by `report.py` and imported on demand.

## 9. Bayesian estimation

### 9.1 Priors and posterior (first results)

| Item | Result |
|---|---|
| Posterior: likelihood + priors by quantiles (beta, inverse gamma, normal, gamma) | equal (1e-6) at 3 points |
| Implied mean and standard deviation of the priors by quantiles | equal (1e-6) |

**Weibull prior by quantiles (does not match, by choice):** the RISE density is not compatible
with the requested quantiles (reported deviation of 0.17 for quantiles 0.5 and 3.0). The package
uses the Weibull that satisfies the quantiles.

### 9.2 What is implemented

Files: `src/msdsge/samplers.py`, `mdd.py`, `optimizers.py`, `diagnostics.py`, `priors_extra.py`;
tests in `tests/test_samplers.py`, `tests/test_mdd.py`, `tests/test_estimation_extras.py`; scripts
`validation/matlab/ref_est.m` and `ref_est_hess.m`; references `validation/ref/est_ms2.mat` and
`est_ms2_hess.mat`.

- Samplers: `mh_sampler` (adaptive Metropolis with the options of the RISE `mh_sampler`; scale and
  covariance by formulas 20, 23 and 24 of Miasojedow, Moulines and Vihola, 2013), `apt_sampler`,
  `dsmh_sampler` (Waggoner, Wu and Zha, 2016), `rrf_sampler` and `smc_sampler` (Herbst and
  Schorfheide, 2014). Fixed seeds, several chains, optional parallelism through `multiprocessing`
  (fork); the result does not depend on the number of processes. Usage:
  `result.sample("mh" | "apt" | "dsmh" | "rrf" | "smc")`.
- MDD: the nine algorithms of the RISE class `mdd` (`mhm`, `swz`, `mueller`, `bridge`, `is`,
  `ris`, `cj`, `laplace`, `laplace_mcmc`) plus `ce` (cross entropy, which is not in RISE). Usage:
  `result.log_mdd(alg)`.
- Optimisers: `bee_gate` (bee colony), `blockwise`, equivalents of `fmincon`, `fminunc`,
  `csminwel`, `fminsearch`, two global optimisers from SciPy and multiple restarts
  (`estimate(optimizer=..., restarts=...)`; also `seed`, `optimizer_options`).
- Priors: cauchy, laplace, logistic, pareto, triangular, truncated_normal, exponential (by mean
  and standard deviation or by quantiles), Dirichlet in the RISE format
  (`dirichlet=[sd_ii, name1, m1, ...]`) and endogenous priors
  (`endogenous_priors=EndogenousPrior(moments, priors)`). `Prior` has a `bounds` field.
- Diagnostics: Gelman-Rubin and Brooks-Gelman, effective sample size, Geweke, autocorrelations,
  running means, trace data and prior × posterior data (the plotting functions return the data),
  `check_optimum`/`mode_curvature`, `identification` (rank of the Jacobian of the moments or of the
  Hessian).

The default test run excludes slow tests (`pyproject.toml`: marker `slow` and
`addopts = "-m 'not slow'"`). The estimation module added 72 tests to the default run (53 s) and 6
slow ones (5 pass, 1 skipped because of pending item 1 below).

### 9.3 Validation on targets with a known answer

Correlated normal and bimodal mixture in 3 dimensions, log of the constant 3.7; development run
with samples larger than those of the tests.

- MDD on the normal: the ten estimators between 3.684 and 3.702. On the mixture: eight between
  3.677 and 3.713; `laplace` gives 3.007 (it sees a single mode, expected bias of log 0.5) and
  `laplace_mcmc` 3.812.
- Samplers: means and variances recovered on both targets; proportion between the modes from 0.49
  to 0.51 in `mh`, `apt`, `dsmh` and `smc`. Built-in MDD: `smc` 3.67 and 3.69; `dsmh` 3.69 on the
  normal.

### 9.4 Validation against RISE

`est_ms2.rs`, `filter_ms_data.csv`, `estimate` with `kf_riccati_tol=0`.

| Item | RISE | package | difference |
|---|---|---|---|
| Mode (7 parameters) | | | less than 4e-8 in each parameter |
| Log posterior at the mode | -394.2422607 | -394.2422607 | 4e-11 |
| Laplace with the Hessian stored by RISE | -413.5557 | -413.5557 | less than 1e-6 |
| Laplace with each one's own Hessian | -413.5557 | -414.1032 | 0.547 |

**What does not match and why.** The Hessian that the RISE `estimate` stores is not the matrix of
second derivatives at the mode: central differences of RISE's own objective (`ref_est_hess.m`)
give a Laplace value of -414.1032, equal to that of the package, and
`utils.hessian.numerical(...,'fd')` gives -414.1027. The difference of 0.547 in the Laplace MDD
reported by RISE comes from there (the standard deviations at the mode also differ, by up to 2.5
times in `a_c_2`). The package keeps the Hessian by central differences.

### 9.5 Deviations and choices of this package

- `rrf_sampler` is the DSMH with the option names and defaults of RRF; RRF has neither a paper nor
  a help text for the algorithm.
- In DSMH, `lambda_1` is raised when the weights of the first stage degenerate. Without this, on
  unbounded support, the MDD came out biased (4.68 against 3.70 in the normal test). The initial
  density is a Student t around the mode, truncated at the bounds.
- In MH, the acceptance target is the midpoint of the `alpha` interval (the RISE rule inside the
  interval is not documented); `c` multiplies the covariance.
- The parametrisation of the additional priors and of `truncated_normal` (mean and standard
  deviation before truncation) is that of the package; it was not compared with RISE.
- Endogenous priors enter without a Jacobian, as a density of the moments; `stdev_moments` uses
  the covariance conditional on the regime.
- `fmincon`, `fminunc` and `csminwel` are functional equivalents (L-BFGS-B and BFGS from SciPy),
  not ports.
- With regime switching, `identification(method="moments")` uses moments per regime, which are not
  directly observed: full rank is an indication, not a proof.

### 9.6 Pending

1. RISE MDDs from a RISE sample: the RISE `mh_sampler` took more than 4 hours for 10 thousand
   iterations on a loaded machine and was interrupted before writing the sample; `mcmc_mdd` did not
   run. The test `test_mdd_from_rise_draws_matches_rise` exists and is skipped while `est_ms2.mat`
   lacks the fields `mh_x` and `mdd` (running `ref_est.m` to the end is enough).
2. Endogenous priors and Dirichlet were not compared numerically with RISE.
3. Formulas of `swz`, `mueller`, DSMH and APT checked by simulation, not equation by equation
   against the papers. Wishart and inverse Wishart priors not implemented (not univariate).
4. Numerical standard errors of the MDD estimators; resuming interrupted chains (`recover`, `save`
   in RISE); `MaxTime`; non-symmetric proposal (`logproppdf`).
5. `multiprocessing` with fork emits a `DeprecationWarning` on Python 3.13 when there are active
   threads.

The literature for this module (18 references with DOI checked on Crossref; Meng and Wong, 1996,
has no DOI) is in `references.bib` and `docs/LITERATURE.md`.

## 10. Occasionally binding constraints

### 10.1 What is implemented

`src/msdsge/occbin.py`: `solve_occbin` (reference regime, reference steady state imposed on all
regimes, `restr_map`), `OccbinSolution.simulate` and `.irf` that honour the constraints (piecewise
linear solution of Guerrieri and Iacoviello, 2015, with one two-state chain per constraint), and
`simulate_through_switch`/`viable_regimes` for the mode `simul_honor_constraints_through_switch`.
Syntax: block `occbin_constraints; [chain:] expr >= expr; end;` (stored in
`ModelSpec.constraints`). The converter takes the `?` lines of RISE to that block (before, they
raised an error). Converted models in `validation/occbin/`, references in
`validation/ref/occbin.mat` (RISE, `validation/matlab/ref_occbin.m`) and `occbin_dynare.mat`
(Dynare 6.4, `ref_occbin_dynare.m` and `occbin_zlb_dynare.mod`), tests in `tests/test_occbin.py`
(12, about 10 s).

**Literature.** Binning and Maih (2017) was read: it describes the formulation by regimes with
endogenous probabilities, not the algorithm of `solve_occbin`. The RISE algorithm has no
documentation (empty help texts) and was inferred by numerical comparison.

### 10.2 RISE behaviour identified

Option `lookahead=False`, the module default:

1. `Tz` of the alternative regime is NaN; only the reference regime is solved.
2. The constraint is handled only if it is violated in the current period by the reference step;
   violations expected only for future periods are ignored.
3. Each constraint stays active for a contiguous block of periods starting at t.
4. The constrained IRF is the path with the shock in the first period, in deviation from the
   steady state.
5. Through the regime switch: the regime is drawn among those that honour the constraint, with the
   row of Q.

Items 2 and 3 differ from the original OccBin, which is available with `lookahead=True`.

### 10.3 Validation

| Validation | Result |
|---|---|
| One constraint (ZLB, `examples/occbin/oneConstraint`): simulation of 300 periods with the RISE shocks (54 at the floor) | equal, identical regimes (test tolerance 5e-8; observed differences of the order of 1e-9, which is the tolerance of the RISE `mfi`) |
| Same, IRFs of 4 shocks, sizes ±5, ±30, ±60 (24 IRFs) | equal (up to 1e-9) |
| Two constraints (`multipleConstraints/ig2017.rs`, 47 variables, 4 regimes): simulation of 300 periods | equal (4e-11), identical regimes |
| Same, IRFs of 6 shocks, ±5 and ±20 (24 IRFs) | 21 equal (up to 8e-11); 3 differ |
| Dynare 6.4 (OccBin), one-constraint model, same shocks, `lookahead=True` | equal (1e-9), same constrained periods |
| Regime switching (`globdynzlb.rs` with `? R>=1`, 2000 periods) | law of motion equal (1e-9); RISE never stays in a regime that violates; switching frequencies compatible with Q |

### 10.4 Does not match

IRFs of the two-constraint model to shocks of -20 deviations in `eps_k`, `eps_p` and `eps_r`
(largest difference of 0.037, 0.126 and 0.0037). By exhaustive search, the sequence of regimes
that RISE uses on impact is, for example, slack collateral in 0 to 2 and ZLB in 3 to 6 (`eps_k`),
or slack collateral for 13 periods and ZLB for 4 (`eps_p`). 62 combinations of updating rule were
tested (accumulate, OccBin, replace; free block, from t, first block, envelope; with and without
reusing the sequence of the previous period) and none reproduces the three cases without losing
the others. The test records the three as a known difference.

### 10.5 Out of scope

Constraints by anticipation of shocks (`simul_frwrd_back_shoot`, `simul_fbs_horizon`,
`simul_anticipate_zero`): in the local RISE, with `solve_shock_horizon=4` and a `?` line in a
one-regime model, forward-back shooting stops with "requires the presence of exogenous variables
to endogenize", and without it the simulation does not honour the constraint (79 of 300 periods
below the floor). Without knowing how the shocks to be endogenised are declared, there is nothing
to reproduce. Also out: orders above the first, filter and estimation with constraints, exit
condition declared in the model file (the field `Constraint.relax` exists only in the API), Dynare
version of the two-constraint model. The RISE examples do not come with Dynare versions; the
one-constraint one was written by hand.

## 11. Optimal policy

### 11.1 What is implemented

`src/msdsge/optimal_policy.py`: `solve_optimal_policy` (commitment, discretion and loose
commitment with regime switching, by functional iteration on the fixed point of Debortoli, Maih
and Nunes, 2014, eqs. 9, 10 and 15), `loss` (conditional and unconditional welfare) and
`optimal_simple_rule` (OSR). Syntax:
`planner_objective(discount=..., commitment=...); expression; end;` (stored in
`ModelSpec.planner`; the parser checks the number of equations). With `commitment` the policy rule
is left out of the model (fewer equations than variables) and `Model.solve` returns an ordinary
`Solution` with the multipliers `MULT_k` (`solve_first_order` hands over to `solve_optimal_policy`
when there is `commitment`); without `commitment` the block is the objective of a simple rule. In
the intermediate case the regimes double (chain `loosecommit`, the slowest: first "plan kept",
then "plan redone"). The converter handles the RISE block `planner_objective{...}`. Tests:
`tests/test_optimal_policy.py` (26). Literature in `docs/LITERATURE.md` (section on optimal
policy) and `references.bib`.

### 11.2 Validated against RISE

`validation/matlab/ref_optimal_policy.m` and `ref_optimal_policy2.m`.

| Case | Largest difference |
|---|---|
| `Canonical.rs`, commitment 0, 0.5 and 1: T and R (variables and multipliers), Q, IRFs | 3e-9 |
| `Canonical_plain.rs` | 3e-10 |
| Fuhrer-Moore (`fm95_ms_lc.rs`), commitment 1, 0.5 and 0: IRFs | 4e-13, 2e-10, 1e-10 |
| Smets-Wouters `usmodel_ms_lc.rs`, 0, 0.5 and 1: T and R | 1.8e-6, 1.3e-6, 4e-9 |
| `usmodel_sr_switch.rs` (83 variables, 2 regimes): T and R | 1.1e-6 |
| `Canonical_osr.rs`: `loss` (V, d, welfare) at the starting point | 1e-9 relative |
| OSR: optimum (0.78329; 6.0040) against (0.78328; 6.0037) in RISE; welfare | -338.51467 in both |
| Variant with the lead coefficient switching with the regime (`op_canL.rs`), `timing="rise"` | 3e-9 |

Canonical NK: closed-form solution under discretion and under commitment (Clarida, Galí and
Gertler, 1999) reproduced to 1e-10; welfare increasing in the probability of commitment;
conditional value checked by Monte Carlo (3%).

### 11.3 Does not match or was not compared, and why

1. `model(linear)` is not accepted by the local RISE: `fm95_ms_lc.rs` and `Canonical_plain.rs` do
   not run as they are; the reference uses copies with `model`.
2. The RISE `loss` fails (a `reshape` error) in every model with multipliers; the loss under
   optimal policy has no reference. Only that of the simple-rule case was compared.
3. Scale: RISE uses the Hessian W in place of W/2; `loss` returns the true value and
   `rise_scale=True` returns that of RISE (twice as large).
4. Lead or lag coefficients switching with the regime: RISE uses the matrices of the current
   regime in the first-order conditions. The default here (`timing="exact"`) uses the lead matrix
   of the regime of t-1 and the lag matrix of the regime of t+1, which is what comes out of the
   derivation (own derivation; to be checked against Blake and Zampolli, 2011). Welfare is higher
   with the exact timing; `timing="rise"` reproduces RISE.
5. Probability of commitment by regime: here `Q = Q_model x [gamma_s, 1-gamma_s]`, with gamma of
   the current regime. The RISE matrix in this case (0.55 starting from "kept" and 0.30 from
   "redone", with gamma of 0.3 and 0.8) looks like a defect and is not reproduced.
6. `usmodel_sr_switch.rs` (83 variables, 2 regimes): T and R equal (see the test), except for the
   multiplier of the definition equation `pinf_target = pinf + ms`, which in RISE comes out with
   the opposite sign (the sign of a multiplier follows the normalisation of the equation; the
   model variables do not change).
7. The stability of Fuhrer-Moore is not compared: there is a unit root in the price level (radius
   1.0000).

### 11.4 Out of scope

Nonlinear models and steady state with non-zero multipliers (raises `NotImplementedError`);
endogenous transition probabilities; anticipated shocks and second order with optimal policy; the
RISE variable `UTIL`; enumeration of multiple discretionary equilibria; estimation of the examples
in `Estimation_examples` (the solution enters the filter, but the mode was not compared); Dynare
version of Fuhrer-Moore (not run); discount that differs across regimes in the unconditional
value. Default discount 0.99 when the block does not give one, as in RISE.

## 12. VAR models and DSGE-VAR

Files: `src/msdsge/var.py`, `src/msdsge/dsge_var.py`, `tests/test_var.py`,
`tests/test_dsge_var.py`, `validation/matlab/ref_var.m`, `validation/matlab/ref_var_ms.m`,
`validation/ref/var_rf.mat`, `validation/ref/var_ms.mat`. `estimation.estimate` accepts
`problem_class=`, used by `dsge_var.estimate_dsge_var`.

### 12.1 What is implemented

- `VAR` (reduced form): constant, trend, exogenous variables; OLS (covariance with T-K, as in
  RISE), maximum likelihood, linear restrictions in the RISE syntax (`'b1(1,dy)=0'`); priors
  `minnesota`, `jeffrey`, `nw`/`normal-wishart`, `inw` (Gibbs) and `sz`; posterior sampler;
  residual bootstrap.
- Regime switching (`MarkovChain`, with the RISE block names: `c`, `a0`, `a1`, `b1`, `s`,
  `s(1)`): Hamilton filter, Kim smoother, EM (reduced form) and direct optimisation (SLSQP, with
  equalities, inequalities and priors on the transition probabilities); known regimes.
- `SVAR` (structural form `A0 y = C d + A_l y(-l) + diag(s) e`, `diag(A0)=1`), with the
  restrictions of Zha's tutorials.
- Identification: Cholesky, short-run and long-run zeros and mixed (Rubio-Ramírez, Waggoner and
  Zha, 2010), signs and zeros with signs (accept-reject, Arias, Rubio-Ramírez and Waggoner, 2018,
  without the importance weights). IRF, GIRF, variance decomposition, historical decomposition,
  structural shocks, autocovariances, forecast and conditional forecast (Waggoner and Zha, 1999).
- DSGE-VAR (Del Negro and Schorfheide, 2004): population moments from `Solution` and from
  `analysis.theoretical_autocovariances`, marginal likelihood with weight lambda (fixed or in the
  parameter `dsge_prior_weight`, as in RISE), posterior of (Phi, Sigma), IRFs with the DSGE
  rotation and joint estimation through `DSGEVARProblem(EstimationProblem)`.

### 12.2 Validation against RISE

| Item | Result |
|---|---|
| `howto.m` (VAR(3), 7 Smets-Wouters variables): coefficients, covariance, likelihood, IRF, variance decomposition (40 horizons and infinity), historical decomposition, shocks, autocovariances, forecast and conditional forecast | equal (1e-9 or better) |
| Exact identification in 6 schemes (short run, long run, mixed, zero at horizon 1), with identified IRF, VD and HD | equal (1e-10) |
| Exogenous variable, no constant, linear restrictions (RISE does restricted least squares, not GLS) | equal (1e-9) |
| Prior `sz`, default hyperparameters, with and without linear restrictions | equal (1e-10) |
| Zha's SVAR (`driver0`; variances in 2 and 3 states): likelihood, posterior, filtered, updated and smoothed probabilities, IRFs per regime, evaluated at the RISE parameter vector | equal (1e-7 or better; posterior 2e-5) |
| Posterior mode of `driver0` found again by the package optimiser | objective 970.93835 in both; parameters within 4e-7 |
| SVAR with variances in 2 states: mode | the package finds objective 807.997; RISE stopped at 902.951 |

Findings about RISE used in the implementation (the code is p-code; everything was inferred from
numbers):

- The `sz` prior is 19 dummy observations (n=3, p=2): n p of the Minnesota type
  (`sigma_i l^L3 / L1`), `L5` repetitions of `diag(sigma)`, n of sum of coefficients (weight `L7`)
  and one of co-persistence (weight `L6`); mean and standard deviation (T-1) over the whole
  sample. `L2` and `L4` do not enter. The reported covariance divides by T + 19.
- With regime switching, the dummy observations are filtered together with the data. The adopted
  order (data, then dummies) reproduces the RISE objective at 298 of 300 test points (2e-5); at
  the other two the difference reaches 0.4 and neither of the two simple orders explains it.
- The filter of the RISE VARs starts from uniform probabilities (1/h), not from the ergodic
  distribution. This is the default of `hamilton_filter`; `pi0='ergodic'` is available.
- The default IRF of `svar` uses `A0^{-1} diag(s)`; that of `rfvar` uses Cholesky.

### 12.3 What does not match or could not be validated

- **Priors `minnesota`, `nw`, `normal-wishart`, `inw`, `jeffrey`:** in the local RISE the first
  four end in an error and `jeffrey` answers "not yet implemented". They follow the literature
  (Minnesota-type variances, normal-Wishart conjugate, Gibbs) and are tested by analytical limits.
  The fields `normal_wishart_eta` and `independent_normal_wishart_eta` of the RISE template have no
  identifiable function and are not used.
- **Prior `sz` away from the defaults:** with `L1, L3, L5, L6, L7` changed the coefficients differ
  by up to 0.06, and with an exogenous variable by 0.03 (0.006 if the co-persistence observation
  carries the mean of the exogenous variable). The mapping of the hyperparameters is confirmed
  only at the default values.
- **Sampler of `sz`:** the means of the coefficients coincide; the covariances drawn by RISE are
  about 10% smaller (degrees of freedom close to T + 31 against T + 19 - K here).
- **Dirichlet prior:** it does not change the RISE objective (difference of 3e-4 with or without
  it). The package applies the Dirichlet density.
- **Beta prior `{0.5,0.1,0.3,'beta'}` of the tutorials:** mean 0.1 and standard deviation 0.3 do
  not define a beta; RISE goes on with degenerate hyperparameters. The package refuses with an
  error.
- **`identification(...,'choleski',order)`:** RISE ignores the order and returns the Cholesky in
  the order of the variables; the package reorders.
- **Sign restrictions:** the matrix returned by the RISE `Rfunc` is a draw with `R R' = Sigma`
  that does not satisfy the restrictions in the order of the named shocks (the filtering happens
  inside `irf`). Properties are compared. `agnostic=false` was not distinguished from `true`.
- **Long-run zero in the VAR(3) with 7 variables:** RISE reports overidentification and ends in an
  error in a scheme that the count of Rubio-Ramírez, Waggoner and Zha gives as exact.
- **DSGE-VAR:** in the local RISE `solve` and `estimate` end in an error (`bvar_dsge`,
  `likelihood_dsge_var`) and `filter` returns the likelihood of the Kalman filter, ignoring the
  weight. There is no numerical comparison; the formula is checked by the identity
  `p(Y) = p(Y|Phi,Sigma) p(Phi,Sigma) / p(Phi,Sigma|Y)` with scipy densities (1e-7) and by the
  limits in lambda.

### 12.4 Pending

- Panel VAR (`panel`, `homogeneity`), endogenous transition probabilities in the VAR,
  `likelihood_smooth_transition_var`, VAR-specific MDD, `print_structural_form`.
- EM only in the reduced form; the SVAR with switching uses direct optimisation, without
  analytical derivatives.
- Models with switching coefficients (`driver1` to `driver5`) did not enter the reference: the
  RISE mode in `driver1` is degenerate (zero transition probability and an explosive regime).
- Historical decomposition and conditional forecast with regime switching use the parameters of a
  fixed regime.
- DSGE-VAR with regime switching uses the moments of one regime, conditional on staying in it.
- `statsmodels` is not installed; there was no cross-check against it.
