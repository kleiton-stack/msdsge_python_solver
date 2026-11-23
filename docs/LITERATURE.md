# LITERATURE

## 2026-11-22: solution and estimation of Markov-switching DSGE models (MS-DSGE) / functional replica of RISE in Python

**Method:** perturbation in regime-switching DSGE models (constant and endogenous probabilities), filters for state-space models with regimes (Kim/GPB, IMM, sigma-point), Bayesian estimation.
**Topic:** reimplement in Python, from the papers, the functionality of the RISE toolbox (closed source, p-code).
**Context:** research software; no specific country or frequency.

### Search terms and sources consulted

- Crossref (`api.crossref.org/works?query.bibliographic=...` and `/works/<DOI>`): confirmation of DOI, journal, volume and pages of every reference in the table.
- OpenAlex (`filter=cites:<id>,from_publication_date:2021-01-01`): forward citations of Foerster et al. (2016), Maih (2015) and Chang, Maih and Tan (2021).
- WebSearch: "Efficient perturbation methods for solving regime-switching DSGE models"; "On Bayesian Filtering for Markov Regime Switching Models"; "Sigma point filters for dynamic nonlinear regime switching models"; "Conditional forecasts in DSGE models"; "Markov-switching DSGE new solution method / filter / determinacy 2023 2024 2025"; "endogenous regime switching DSGE perturbation time-varying transition probabilities"; "Kim filter approximation accuracy collapsing alternative"; "Python OR Julia package Markov-switching DSGE"; "DSGE.jl regime switching"; "Foerster perturbation code supplement"; "Maih Waggoner time-varying coefficients and transition matrices".
- PDFs downloaded and read in this session (text extracted with `pdftotext`):
  - Maih (2015), Norges Bank WP 1/2015 (49 pp.), from the Norges Bank website.
  - Maih, Hashimzade, Kirsanov and Kirsanova (2026), Glasgow WP 2026-01.
  - Hashimzade, Kirsanov, Kirsanova and Maih (2024), Norges Bank WP 8/2024 (sections 2.2 and 2.3).
- Software pages and APIs: PyPI JSON (`dsgelab`), GitHub API (`jmaih/RISE_toolbox`, `angusmoore/RSDSGE.jl`, `FRBNY-DSGE/DSGE.jl`, `thorek1/MacroModelling.jl`), Dynare 7.2 manual, Quantitative Economics supplemental material page, Zenodo.

Convention for the "Type" column: **Published** = peer-reviewed journal or book; **WP** = working paper. When a published version exists, it is the one cited.

### Table 1. Original methods

| Reference (DOI/link) | Type | What it contributes | Implication for the project | Package |
|---|---|---|---|---|
| Maih (2015), "Efficient perturbation methods for solving regime-switching DSGE models", Norges Bank WP 1/2015. [page](https://www.norges-bank.no/en/news-events/publications/Working-Papers/2015/12015/), SSRN DOI [10.2139/ssrn.2602453](https://doi.org/10.2139/ssrn.2602453) | WP (no journal version found) | Generic system with regimes, endogenous probabilities and anticipated shocks; perturbation without partitioning parameters; functional iteration and Newton via coupled Sylvester; orders 2 and 3; MSS | **Central reference for the solver.** See "Technical summary". Appendix C has test models and parametrisations; Table 2 has the number of real and stable solutions of each | RISE (`mfi`, `mnk`, `mn`) |
| Foerster, Rubio-Ramírez, Waggoner and Zha (2016), Quantitative Economics 7(2), 637-669. [10.3982/QE596](https://doi.org/10.3982/qe596) | Published | Partitioned perturbation; all 1st-order solutions via Gröbner bases; selection by MSS; 1st order may fail certainty equivalence | Special case of Maih's approach (single approximation point). Source of the RBC and NK test models. Gröbner does not scale, do not implement as the default | Supplement with code on the QE page (zip not inspected) |
| Farmer, Waggoner and Zha (2009), JET 144(5), 1849-1867. [10.1016/j.jet.2009.05.004](https://doi.org/10.1016/j.jet.2009.05.004) | Published | Characterises equilibria in forward-looking linear MSRE models; determinacy | Underpins the uniqueness tests; shows that the regime-by-regime condition is not enough | n/a |
| Farmer, Waggoner and Zha (2011), JEDC 35(12), 2150-2166. [10.1016/j.jedc.2011.08.005](https://doi.org/10.1016/j.jedc.2011.08.005) | Published | MSV solutions by Newton; comparison with FP and SW | Benchmark: three parametrisations with 1, 2 and 4 stable solutions. Assumes a lead matrix that depends only on the current regime, which Maih (2015, §3.2.1) shows to be restrictive | RISE includes `fwz` |
| Farmer, Waggoner and Zha (2010), AER 100(1), 608-617. [10.1257/aer.100.1.608](https://doi.org/10.1257/aer.100.1.608) | Published | Comment on Davig-Leeper: the generalised Taylor principle does not guarantee uniqueness | Do not label a solution "determinate" only because it satisfies an eigenvalue condition | n/a |
| Davig and Leeper (2007), AER 97(3), 607-635. [10.1257/aer.97.3.607](https://doi.org/10.1257/aer.97.3.607) | Published | Long-run Taylor principle with regimes | Canonical test example (NK with switching monetary policy) | n/a |
| Costa, Fragoso and Marques (2005), *Discrete-Time Markov Jump Linear Systems*, Springer. [10.1007/b138575](https://doi.org/10.1007/b138575) | Published (book) | Theory of mean-square stability (MSS) of Markov jump linear systems, ch. 3 | Defines the stability test: spectral radius of the augmented matrix below 1 | n/a |
| Cho (2016), RED 21, 182-200. [10.1016/j.red.2015.03.001](https://doi.org/10.1016/j.red.2015.03.001) | Published | Forward method for MSRE; sufficient conditions for determinacy in MSS | Source of the test models `cho2014_EIS` and `cho2014_MP` (lead matrix depends on the future regime). Candidate determinacy diagnostic | n/a |
| Cho and Moreno (2011), JEDC 35(3), 257-272. [10.1016/j.jedc.2010.09.007](https://doi.org/10.1016/j.jedc.2010.09.007) | Published | Forward method as a solution refinement | Basis of the Cho (2016) algorithm | n/a |
| Svensson and Williams (2005), "Monetary policy with model uncertainty: distribution forecast targeting", NBER WP 11733. [10.3386/w11733](https://doi.org/10.3386/w11733) | WP (no journal version found) | Markov jump-linear-quadratic; functional iteration ("SW" algorithm) | Precursor of functional iteration; MSS as the stability concept | n/a |
| Svensson and Williams (2008), FRB St. Louis Review 90(4), 275-294. [10.20955/r.90.275-294](https://doi.org/10.20955/r.90.275-294) | Published | Optimal policy in MJLQ | Relevant for the optimal policy module | n/a |
| Blake and Zampolli (2011), JEDC 35(10), 1626-1651. [10.1016/j.jedc.2011.04.001](https://doi.org/10.1016/j.jedc.2011.04.001) | Published | Optimal policy (commitment and discretion) in MSRE | Same, optimal policy module | n/a |
| Debortoli, Maih and Nunes (2014), Macroeconomic Dynamics 18(1), 175-198. [10.1017/S1365100512000326](https://doi.org/10.1017/s1365100512000326) | Published | Loose commitment in medium-scale models | Algorithm of the optimal policy module with stochastic reoptimisation (a 2-regime case) | RISE |
| Maih (2010), "Conditional forecasts in DSGE models", Norges Bank WP 7/2010. [10.2139/ssrn.1636743](https://doi.org/10.2139/ssrn.1636743) | WP | Conditional forecasting with anticipated information and uncertainty (hard/soft conditions) | Basis of the anticipated shocks (state with ε from t to t+k) and of the conditional forecasting module | RISE |
| Kim (1994), J. Econometrics 60(1-2), 1-22. [10.1016/0304-4076(94)90036-1](https://doi.org/10.1016/0304-4076(94)90036-1) | Published | Filter with h² to h collapsing and probability smoother | Implement as an alternative filter (equivalent to GPB(2)); the probability smoother is the one used in RISE | `statsmodels` has versions for MS-AR, not for the general state space |
| Kim and Nelson (1999), *State-Space Models with Regime Switching*, MIT Press. [10.7551/mitpress/6444.001.0001](https://doi.org/10.7551/mitpress/6444.001.0001) | Published (book) | Reference treatment of the Kim filter and Gibbs sampling | Numerical validation reference | n/a |
| Hamilton (1989), Econometrica 57(2), 357-384. [10.2307/1912559](https://doi.org/10.2307/1912559) | Published | Hamilton filter for regimes | Bayesian update of the regime probabilities | `statsmodels` |
| Blom and Bar-Shalom (1988), IEEE TAC 33(8), 780-783. [10.1109/9.1299](https://doi.org/10.1109/9.1299) | Published | IMM filter: mixing before prediction, h Kalman filters per period | **Default filter in RISE**, so it must be the package default | n/a |
| Blom (1984), 23rd IEEE CDC, 656-658. [10.1109/CDC.1984.272089](https://doi.org/10.1109/cdc.1984.272089) | Published (proceedings) | First version of the IMM | Historical citation | n/a |
| Hashimzade, Kirsanov, Kirsanova and Maih (2026), "Filtering and Smoothing in State-Space Models with Multiple Regimes", JBES. [10.1080/07350015.2026.2656466](https://doi.org/10.1080/07350015.2026.2656466). WP version: Norges Bank WP 8/2024, [10.2139/ssrn.4723734](https://doi.org/10.2139/ssrn.4723734), arXiv 2402.08051 | Published (online Jul 2026, no volume yet). The text read was the WP | GPB(N) and IMM(N) families; Kim-Nelson = GPB(2); canonical IMM is faster and no less accurate; stable state smoother | Algorithms 1 to 3 of the WP are the specification of the filtering and smoothing module | RISE |
| Maih, Hashimzade, Kirsanov and Kirsanova (2026), "Markov-Switching DSGE Modeling in RISE", Glasgow WP 2026-01. [PDF](https://www.gla.ac.uk/media/Media_1242001_smxx.pdf) | WP (chapter planned for an Edward Elgar handbook) | Full RISE workflow: model, solution, IMM, mode, RWMH, MDD, counterfactuals | Defines the API and the defaults to replicate. See "Technical summary" | RISE |
| Binning and Maih (2015), "Sigma point filters for dynamic nonlinear regime switching models", Norges Bank WP 10/2015. [10.2139/ssrn.2616801](https://doi.org/10.2139/ssrn.2616801) | WP | Extends the UKF, DDF and CKF to nonlinear models with regimes | Filter for higher-order solutions. PDF not read in this session | RISE |
| Binning and Maih (2017), "Modelling occasionally binding constraints using regime-switching", Norges Bank WP 23/2017. [10.2139/ssrn.3073753](https://doi.org/10.2139/ssrn.3073753) | WP | OBC (ZLB, collateral, wage rigidity) as regimes with endogenous probabilities | Use case for endogenous probabilities; source of test models. PDF read on 2026-11-22 (see the section on occasionally binding constraints, at the end) | RISE |
| Binning and Maih (2016), "Implementing the zero lower bound in an estimated regime-switching DSGE model", Norges Bank WP 3/2016. [page](https://www.norges-bank.no/en/news-events/publications/Working-Papers/2016/32016/) | WP | ZLB with two steady states | Example of regime-specific steady states | RISE |
| Barthélemy and Marx (2017), JEDC 77, 1-25. [10.1016/j.jedc.2017.01.011](https://doi.org/10.1016/j.jedc.2017.01.011) | Published | Perturbation with state-dependent probabilities; local existence and uniqueness | Theoretical foundation for endogenous probabilities, which Maih handles without a stability criterion | n/a |
| Benigno, Foerster, Otrok and Rebucci (2025), Quantitative Economics 16(1), 1-47. [10.3982/QE2038](https://doi.org/10.3982/qe2038) | Published | Perturbation with endogenous switching (collateral constraint); Bayesian estimation | The 1st order coincides with that of an exogenous model evaluated at the steady-state transition matrix; the effects of endogeneity show up from the 2nd order onwards. Usable as a test | Replication package: [10.5281/zenodo.14026657](https://doi.org/10.5281/zenodo.14026657) |
| Chang, Maih and Tan (2021), JEDC 133, 104235. [10.1016/j.jedc.2021.104235](https://doi.org/10.1016/j.jedc.2021.104235) | Published | Threshold switching with endogenous feedback; dedicated filter | Later extension (not the core). Requires a filter with an autoregressive latent factor | RISE |
| Chang, Choi and Park (2017), J. Econometrics 196(1), 127-143. [10.1016/j.jeconom.2016.09.005](https://doi.org/10.1016/j.jeconom.2016.09.005) | Published | Threshold regime model with a latent factor | Econometric basis of Chang, Maih and Tan | n/a |
| Andreasen, Fernández-Villaverde and Rubio-Ramírez (2018), REStud 85(1), 1-49. [10.1093/restud/rdx037](https://doi.org/10.1093/restud/rdx037) | Published | Pruned system for higher orders; closed-form moments | Simulation at order 2 or above must be pruned; apply regime by regime | Dynare |
| Levintal (2017), JEDC 80, 1-16. [10.1016/j.jedc.2017.04.007](https://doi.org/10.1016/j.jedc.2017.04.007) | Published | Perturbation up to 5th order with states in a single block | Single-block notation used by Maih (2015, §3) | n/a |
| Kamenik (2005), Computational Economics 25, 167-187. [10.1007/s10614-005-6280-y](https://doi.org/10.1007/s10614-005-6280-y) | Published | Generalised Sylvester for higher orders without regimes | Only useful for the h = 1 case; with regimes use TFQMR or Kronecker | Dynare++ |
| Freund (1993), SIAM J. Sci. Comput. 14(2), 470-482. [10.1137/0914029](https://doi.org/10.1137/0914029) | Published | TFQMR | Default iterative solver in RISE for the coupled Sylvester equation | `scipy.sparse.linalg.tfqmr` |
| Waggoner, Wu and Zha (2016), J. Econometrics 192(2), 406-420. [10.1016/j.jeconom.2016.02.007](https://doi.org/10.1016/j.jeconom.2016.02.007) | Published | DSMH sampler for multimodal posteriors | Alternative sampler to RWMH; useful because MS-DSGE posteriors are multimodal | Authors' C code (not verified) |
| Herbst and Schorfheide (2014), JAE 29(7), 1073-1098. [10.1002/jae.2397](https://doi.org/10.1002/jae.2397) | Published | SMC for DSGE | Alternative sampler; delivers the MDD as a by-product | n/a |
| Bognanni and Herbst (2018), JAE 33(1), 126-140. [10.1002/jae.2582](https://doi.org/10.1002/jae.2582) | Published | SMC in multi-equation MS models | Evidence that SMC copes well with label switching and multimodality | n/a |
| Bianchi (2013), REStud 80(2), 463-490. [10.1093/restud/rds032](https://doi.org/10.1093/restud/rds032) | Published | MS-DSGE estimated with Gibbs; belief counterfactuals | Reference application to validate smoothed probabilities | n/a |
| Bianchi (2016), J. Econometrics 190(1), 79-99. [10.1016/j.jeconom.2015.08.004](https://doi.org/10.1016/j.jeconom.2015.08.004) | Published | Analytical formulas for expectations and uncertainty in MS models | Specification of regime-conditional forecasts and moments | n/a |
| Bianchi and Melosi (2017), AER 107(4), 1030-1058. [10.1257/aer.20160186](https://doi.org/10.1257/aer.20160186) | Published | MS-DSGE with monetary and fiscal policy regimes | Reference application | n/a |
| Bianchi and Ilut (2017), RED 26, 113-139. [10.1016/j.red.2017.02.011](https://doi.org/10.1016/j.red.2017.02.011) | Published | Monetary/fiscal mix with regimes | Reference application | n/a |
| Liu, Waggoner and Zha (2011), Quantitative Economics 2(2), 251-301. [10.3982/QE71](https://doi.org/10.3982/qe71) | Published | Medium-scale MS-DSGE; model comparison by MDD | Protocol for comparing regime specifications | n/a |
| Foerster (2016), IER 57(1), 211-230. [10.1111/iere.12153](https://doi.org/10.1111/iere.12153) | Published | Expectation effects of regime change | NK test model | n/a |
| Sims and Zha (2006), AER 96(1), 54-81. [10.1257/000282806776157678](https://doi.org/10.1257/000282806776157678) | Published | MS-SVAR | Specification of the VAR-with-regimes module | Dynare (MS-SBVAR) |
| Guerrieri and Iacoviello (2015), JME 70, 22-38. [10.1016/j.jmoneco.2014.08.005](https://doi.org/10.1016/j.jmoneco.2014.08.005) | Published | OccBin | Benchmark for OBC via regimes | Dynare |
| Durbin and Koopman (2012), *Time Series Analysis by State Space Methods*, 2nd ed., OUP. [10.1093/acprof:oso/9780199641178.001.0001](https://doi.org/10.1093/acprof:oso/9780199641178.001.0001) | Published (book) | Smoother by backward recursion | Basis of the state smoother of Hashimzade et al. | `statsmodels` (single regime) |
| Karaboga and Basturk (2007), J. Global Optimization 39, 459-471. [10.1007/s10898-007-9149-x](https://doi.org/10.1007/s10898-007-9149-x) | Published | Artificial bee colony | RISE's global optimiser `bee_gate` for the mode | n/a |
| Julier and Uhlmann (2004), Proc. IEEE 92(3), 401-422. [10.1109/JPROC.2003.823141](https://doi.org/10.1109/jproc.2003.823141) | Published | Unscented filter | Basis of the sigma-point filters | `filterpy` (not verified) |

### Table 2. Recent advances (2021 to 2026)

| Reference (DOI/link) | Type | What it contributes | Implication for the project | Package |
|---|---|---|---|---|
| Cho (2021), JEDC 127, 104115. [10.1016/j.jedc.2021.104115](https://doi.org/10.1016/j.jedc.2021.104115) | Published | Complete classification of MSRE models into determinate, indeterminate and no stable solution (MSS), via the forward solution | Determinacy diagnostic that can be implemented without enumerating all solutions. Complements Maih, who only finds one solution per initial guess | n/a |
| Barthélemy, Cho and Marx (2024), RED 54, 101240. [10.1016/j.red.2024.101240](https://doi.org/10.1016/j.red.2024.101240) | Published | Unified necessary and sufficient conditions; determinacy in the boundedness sense is not equivalent to determinacy in MSS | The solver report must state which stability concept was used | n/a |
| Barthélemy and Marx (2019), Quantitative Economics 10(1), 353-385. [10.3982/QE673](https://doi.org/10.3982/qe673) | Published | Determinacy in models with policy switching (bounded solutions) | Same | n/a |
| Cho and Moreno (2024, online), JMCB 58(2), 421-448. [10.1111/jmcb.13204](https://doi.org/10.1111/jmcb.13204) | Published | Determinacy with monetary/fiscal switching and the ZLB | Test case for the diagnostic | n/a |
| Albertini and Moyen (2024), Computational Economics 64(6), 3645-3682. [10.1007/s10614-024-10570-z](https://doi.org/10.1007/s10614-024-10570-z) | Published | General method for solving DSGE models with regimes | Competing method; read before fixing the solver. Only abstract and metadata verified | not verified |
| Benigno et al. (2025), QE (row in Table 1) | Published | Estimated endogenous switching | See Table 1 | Zenodo |
| Foerster and Matthes (2022), IER 63(4), 1829-1859. [10.1111/iere.12585](https://doi.org/10.1111/iere.12585) | Published | Regimes with Bayesian learning by agents | Extension outside the initial scope (imperfect information about the regime) | n/a |
| Bianchi and Melosi (2016), IER 57(2), 717-756. [10.1111/iere.12174](https://doi.org/10.1111/iere.12174) | Published | Beliefs and learning in general equilibrium with regimes | Same | n/a |
| Hashimzade et al. (2026), JBES (row in Table 1) | Published | IMM versus Kim-Nelson | Reason for IMM as the default | RISE |
| Li (2026), Econometric Theory. [10.1017/S0266466626100723](https://doi.org/10.1017/s0266466626100723) | Published (online Nov 2026) | Asymptotic properties of the approximate maximum likelihood estimator in state-space models with regimes | Addresses the validity of inference with an approximate likelihood (collapsing). Only title and metadata verified; read before writing the inference section | n/a |
| Pouzo, Psaradakis and Sola (2022), Econometrica 90(4), 1681-1710. [10.3982/ECTA17249](https://doi.org/10.3982/ecta17249) | Published | MLE with covariate-dependent transition probabilities, under misspecification | Theoretical support for time-varying probabilities in the likelihood | n/a |
| Herbst and Schorfheide (2019), J. Econometrics 210(1), 26-44. [10.1016/j.jeconom.2018.11.003](https://doi.org/10.1016/j.jeconom.2018.11.003) | Published | Tempered particle filter | Exact (in simulation) alternative for higher order, to validate the sigma-point filter | n/a |
| Aruoba, Cuba-Borda, Higa-Flores, Schorfheide and Villalvazo (2021), RED 41, 96-120. [10.1016/j.red.2020.12.003](https://doi.org/10.1016/j.red.2020.12.003) | Published | Piecewise-linear approximation and filtering with OBC | Benchmark for OBC | n/a |
| Ivashchenko (2016), "Estimation and filtering of nonlinear MS-DSGE models", SSRN. [10.2139/ssrn.2781924](https://doi.org/10.2139/ssrn.2781924) | WP | Gaussian filters for 2nd-order MS-DSGE | Alternative to sigma-point. Only metadata verified | n/a |
| Lhuissier and Tripier (2021), Quantitative Economics 12(4), 1139-1170. [10.3982/QE1298](https://doi.org/10.3982/qe1298) | Published | MS-DSGE with financial regimes | Recent application | n/a |
| Bianchi, Melosi and Rottner (2021), JME 124, 107-122. [10.1016/j.jmoneco.2021.10.003](https://doi.org/10.1016/j.jmoneco.2021.10.003) | Published | ZLB and deflationary bias with regimes | Recent application | n/a |
| Maih, Mazelis, Motto and Ristiniemi (2021), J. Macroeconomics 70, 103376. [10.1016/j.jmacro.2021.103376](https://doi.org/10.1016/j.jmacro.2021.103376) | Published | Asymmetric rules with the ELB via regimes | Application done in RISE; candidate integration test | RISE |
| Bianchi, Faccini and Melosi (2023), QJE 138(4), 2127-2179. [10.1093/qje/qjad027](https://doi.org/10.1093/qje/qjad027) | Published | Persistent fiscal inflation | Recent application from the policy-regime literature | n/a |
| Kocięcki and Kolasa (2023), J. Econometrics 236(2), 105477. [10.1016/j.jeconom.2023.105477](https://doi.org/10.1016/j.jeconom.2023.105477) | Published | Global identification in DSGE | Relevant for the identification module; does not cover regimes | n/a |
| Shi (2026), "Semi-Markov DSGE models: theory, solution, and estimation", Research Square. [10.21203/rs.3.rs-10313432/v1](https://doi.org/10.21203/rs.3.rs-10313432/v1) | Preprint, not peer reviewed | Non-geometric regime duration | Record only; do not use as main evidence | n/a |

### Table 3. Existing software

| Software (link) | Language | Latest release / activity (checked on 2026-11-22) | What it does | Limitations |
|---|---|---|---|---|
| [RISE toolbox](https://github.com/jmaih/RISE_toolbox) | MATLAB | Release tagged `v2019` ("Legacy Release (2019)") published on 2026-03-29; last commit on 2026-09-27 | Complete functional reference | Core in p-code; requires MATLAB |
| [dsgelab](https://pypi.org/project/dsgelab/) (GitHub `GustavoHM/dsgelab`) | Python ≥ 3.11 | 1.0.0 on 2026-11-20; first release (0.5.0) on 2026-10-18 | MS up to 2nd order by partitioned perturbation, "all solutions", MSS, IRF by regime, pruning; likelihood by the Kim filter; estimation of parameters and probabilities | According to its own README: exogenous probabilities only, stops at 2nd order, Kim filter on the 1st-order solution. No IMM, anticipated shocks or regime-specific steady states in the description. The package is one month old; the README claims were not tested |
| [RSDSGE.jl](https://github.com/angusmoore/RSDSGE.jl) | Julia | No release; "experimental" badge; last commit on 2026-09-27 | Solves DSGE models with regimes | README shows only the solution; method and estimation not verified |
| [DSGE.jl](https://github.com/FRBNY-DSGE/DSGE.jl) (FRBNY) | Julia | v1.3.0 on 2021-11-23; push on 2026-10-30 | Exogenous, unanticipated regimes (gensys per regime), estimation with MH and SMC | According to the documentation consulted via search, it is not MS with expectations over future regimes; I did not read the current page directly |
| [Dynare 7.2](https://www.dynare.org/manual/the-model-file.html) | MATLAB/Octave | Version 7.2 manual | "Markov-switching SBVAR" section (Sims, Waggoner and Zha, 2008). The `markov_switching` command accepts options "in case of estimating a MS-DSGE model" and points to `tests/ms-dsge/test_ms_dsge.mod` | I found no documented command in the manual for solving or estimating MS-DSGE |
| Code of Foerster et al. (2016): [QE supplemental material](https://www.econometricsociety.org/publications/quantitative-economics/online/supplemental-material/2016/07/01/Perturbation-methods-for-Markovswitching-dynamic-stochastic-general-equilibrium-models) | not verified | 2016 | Replication of the paper | There is a zip on the page; contents and language not inspected |
| Replication of Benigno et al. (2025): [10.5281/zenodo.14026657](https://doi.org/10.5281/zenodo.14026657) | not verified | 2024-11-01, CC BY 4.0 | Solution and estimation with endogenous switching | Language not stated on the page |
| [MacroModelling.jl](https://github.com/thorek1/MacroModelling.jl), JOSS [10.21105/joss.05598](https://doi.org/10.21105/joss.05598) | Julia | v0.2.2 on 2026-10-29 | Single-regime DSGE, higher orders | I found no MS support |

---

## Technical summary of the algorithms

Equation and section numbers refer to the text of each paper. The notation is abbreviated; check the PDF before coding each block.

### A. Maih (2015), Norges Bank WP 1/2015

**A.1 Form of the system (§2.3, eqs. 1 to 3).**
- Problem: `E_t Σ_{r'=1..h} π_{r,r'}(I_t) d̃_r(v) = 0`, with `r = r_t`, `r' = r_{t+1}`. Defining `d_{r,r'} ≡ π_{r,r'}(I_t) d̃_r`, the objective is `E_t Σ_{r'} d_{r,r'}(v) = 0` (eq. 2). The probabilities enter **inside** the derivatives, so all the `A` matrices below are already probability-weighted.
- Vector `v` (eq. 1): `[b_{t+1}(r'), f_{t+1}(r'), s_t(r), p_t(r), b_t(r), f_t(r), p_{t-1}, b_{t-1}, ε_t, θ_{r'}]`. Typology: `s` static, `p` predetermined, `b` "both", `f` forward-looking. The parser classifies automatically.
- End-of-period dating (§2.2): `K_t = (1-δ)K_{t-1} + I_t`. The paper argues against stacking periods (Klein, Sims, Schmitt-Grohé and Uribe), because the future variable belongs to another regime.
- Restriction on endogenous probabilities (§2.3): the parameters that affect them do not switch and the variables that enter them have a unique steady state.
- States (§2.3.1): `z_t = [p_{t-1}, b_{t-1}, σ, ε_t, ε_{t+1}, ..., ε_{t+k}]`, `n_z = n_p + n_b + (k+1) n_ε + 1`. The anticipated shocks are the structural shocks themselves with k leads, without changing equations (unlike "news shocks").
- Solution sought (eq. 3): `y_t(r) = T^r(z_t)`. Law of motion of the states: `z_{t+1} = h^r(z_t) + u z_t`, with the random `u` defined in eq. 4.
- Future-regime parameters that appear with a lead (eq. 5): `θ_{r'} = θ̄_r + σ θ̂_{r'}`. Proposition 1 justifies treating them as an auxiliary variable whose 1st-order expansion only has a term in `σ`. There is never any perturbation with respect to the current-regime parameters.

**A.2 Order zero: approximation point (§3.1).** Three options: (i) ergodic mean, as in Foerster et al. (RISE option `unique`); (ii) **regime-specific steady state** (RISE default), solving `d̃_r(·; θ_r) = 0` for each `r`; (iii) imposed point (`imposed`). In (i) and (iii) the residual `d̆_r` can be non-zero and is carried over to the 1st order.

**A.3 First order (§3.2).**
- General condition (eq. 9): `d̆_r + Σ_{r'} d_v^{r,r'} E_t v_z = 0`, with `v_z = a0_z + a1_z u` (eq. 10, appendix A.1).
- **Quadratic matrix equation with regimes (eq. 11):**
  `A0_r T_x^r + A−_r + Σ_{r'} A+_{r,r'} T_x^{r'} λ_x T_x^r = 0`, for `r = 1..h`,
  where `T_x^r` is the block of `T_z^r` on the endogenous states, `λ_x` selects the `p` and `b` rows, `A0_r = Σ_{r'} A0_{r,r'}`, `A−_r = Σ_{r'} [d_{p−}, d_{b−}]` and `A+_{r,r'} = [0, 0, d_{b+}^{r,r'}, d_{f+}^{r,r'}]` (eq. 12).
- `A+` depends on `(r, r')`. Farmer, Waggoner and Zha (2011) assume `A+_{r,r'} = A+_{r,r}`; Cho (2016) is the general case. The FWZ algorithm gives the wrong answer in the `cho2014_EIS` model (§6.2).
- Uncertainty term `T_σ` (§3.2.2, eq. 13): stacked linear system of dimension `h·n_y`, with blocks `A_σ^r + A+_{r,r}` on the diagonal and `A+_{r,r'}` off it, and right-hand side `d̆_r + Σ_{r'} d_{θ+}^{r,r'} θ̂_{r,r'}`. Certainty equivalence only breaks at 1st order if (1) future-regime parameters appear in the current system or (2) the approximation point is not the steady state of the regime.
- Shocks (§3.2.3): `U_r ≡ (Σ_{r'} A+_{r,r'} T_x^{r'}) λ_x + A0_r` (eq. 14); contemporaneous `T_{ε0}^r = −U_r^{-1} Σ_{r'} d_{ε0}^{r,r'}` (eq. 15); anticipated by the recursion `T_{εk}^r = −U_r^{-1} Σ_{r'} A+_{r,r'} T_{ε(k−1)}^{r'}` (eq. 16).

**A.4 Higher orders (§3.3 to §3.5, appendices A.2 and A.3).**
- 2nd order (eq. 20): `A_zz^r + Σ_{r'} A+_{r,r'} T_zz^{r'} C_zz^r + U_r T_zz^r = 0`, with `C_zz^r = (h_z^r)^{⊗2} + E u^{⊗2}` and `A_zz^r = Σ_{r'} d_vv^{r,r'} E_t v_z^{⊗2}`.
- 3rd order (eq. 22): same structure, with `C_zzz^r = P(h_z^r ⊗ E u^{⊗2}) + E u^{⊗3} + (h_z^r)^{⊗3}`.
- Order p (§3.5): general formula by index partitions. States in a single block (Levintal) avoid cross blocks.
- Moments of `u`: `E u^{⊗p}` only has the lower block non-zero, equal to `vec(M_p) m_σ^{⊗p}` (footnote 37).
- All orders ≥ 2 are linear in the unknowns and use the same `U_r` and `A+` as the 1st order.

**A.5 Generalised coupled Sylvester equation (§4, eq. 24).** After pre-multiplying by `U_r^{-1}`: `X_i + Σ_{j=1..h} F_ij X_j G_ij = C_i`.
- Compression of symmetric columns with matrices `K` (selection) and `E` (duplication): number of unique columns `C(c_x + p − 1, p)` (footnote 20).
- Direct approach (§4.1): vectorise and invert the block matrix `I + G_ij' ⊗ F_ij`. High cost.
- Iterative approach (§4.2): TFQMR (Freund, 1993), using only matrix-vector products; it is the RISE default.

**A.6 Algorithms for eq. 11 (§5.2).**
- **Functional iteration (eq. 25, Algorithm 1):** `T_x^{r,(k)} = −[A0_r + (Σ_{r'} A+_{r,r'} T_x^{r',(k−1)}) λ_x]^{-1} A−_r`. Names in RISE: `mfi_full` and `mfi` (exploits static variables). Theorem 2 gives a sufficient condition for local convergence: `τ a+ (ϑ + δ) < 1`. Functional iteration may fail to find solutions that exist (the `fwz2011_1` case and Smets-Wouters in Table 2).
- **Newton (eqs. 26 to 29, Algorithm 2):** residual `W_r(T) = T_x^r + U_r^{-1} A−_r` (eq. 26). With `L+^{r,r'} = U_r^{-1} A+_{r,r'}` (eq. 27) and `L−^r = −λ_x T_x^r` (eq. 28), the step solves `W_r + Δ_r − (Σ_{r'} L+^{r,r'} Δ_{r'}) L−^r = 0` (eq. 29), which is eq. 24 with `F_ij = L+`, `G_ij = −L−`, `C_i = −W_i`. Variants: `mnk_full`/`mnk` (direct Kronecker) and `mn_full`/`mn` (TFQMR).
- Performance (Table 1): `mnk_full` is the fastest in small models (2 to 6 variables); `mn` is the fastest in Smets-Wouters with 40 variables (about 7 times faster than `mnk` and 39 times faster than `mnk_full`). Footnote 35 suggests a line search in the Newton step.
- **Initialisation (§5.4):** (1) `T = 0`; (2) "backward", `T_x^r = −(A0_r)^{-1} A−_r`, the RISE default; (3) random, drawing `A+ T` from a scaled normal, used to search for alternative solutions.
- The algorithms return **one** real solution, which depends on the guess. With 100 random guesses (§6.3, Table 2) Newton finds, for example, 1, 2 and 4 stable solutions in the three parametrisations of FWZ (2011), 3 real solutions with 1 stable in the NK model of FRWZ (2013) and 6 real with 1 stable in the NK model with habit.

**A.7 Stability (§5.3, eq. 30).** With constant `Q`, the solution is MSS if and only if all eigenvalues of `Υ = (Q ⊗ I_{n_x²}) · blkdiag(λ_x T_x^1 ⊗ λ_x T_x^1, ..., λ_x T_x^h ⊗ λ_x T_x^h)` lie inside the unit circle. Footnote 29 gives the block form `p_{i,j} (λ_x T_x^j ⊗ λ_x T_x^j)`. Caveats from the paper itself: (i) with endogenous probabilities the MSS concept does not apply; (ii) stability of the 1st order does not guarantee that of higher orders; (iii) checking MSS is usually the most expensive step (an `h n_x² × h n_x²` matrix).

**A.8 Ready-made tests (§6.1 and appendix C).** Ten problems with equations and parametrisations: `cho2014_EIS`, `cho2014_MP`, `frwz2013_nk_1`, `frwz2013_nk_2`, `frwz2013_nk_hab`, `frwz2014_rbc`, `fwz2011_1/2/3`, `smets_wouters`. §6.4 prints the 3rd-order solution of the FRWZ RBC model (steady state of regime 1: C = 2.082588; K = 22.150375; Z = 1.007058).

### B. Glasgow tutorial (Maih, Hashimzade, Kirsanov and Kirsanova, 2026)

- **Three-matrix form (§2.6, eq. 1.10; §3.3, eqs. 1.12 and 1.13):** `A−^{(s_t)} α_{t−1} + A0^{(s_t)} α_t + A+^{(s_t)} E_t α_{t+1} + B^{(s_t)} ε_t = 0`, with `E_t α_{t+1} = Σ_j Pr(s_{t+1} = j | s_t = i) E_t[α_{t+1} | s_{t+1} = j]`. Regime-specific solution `α_t = T^{(s_t)} α_{t−1} + R^{(s_t)} ε_t`. Default solver: `mfi`. The text states that the resulting solution is MSS.
- **Transition matrix convention:** `Q_ij = Pr(s_{t+1} = j | s_t = i)`, rows sum to one. Several independent chains: the regimes are the Cartesian product of the states (two binary chains give 4 regimes; index `2(pol − 1) + vol` in the example of §6.3). Parameters are named `name_chain_state` and probabilities `chain_tp_i_j`.
- **IMM filter, the default (§4.1):** (1) mixing: `μ^{i|j}_{t−1|t−1} = Q_ij μ^i_{t−1|t−1} / Σ_k Q_kj μ^k_{t−1|t−1}`, with mean and covariance mixed per regime `j` (including the spread-of-means term); (2) one Kalman step per regime, returning the predictive density `Λ^j_t`; (3) update `μ^j_{t|t} ∝ Λ^j_t Σ_i Q_ij μ^i_{t−1|t−1}`, whose denominator `ℓ*_t = Σ_k Σ_m Λ^m_t Q_km μ^k_{t−1|t−1}` is the likelihood increment; (4) weighted combination for the expected updated variables; (5) recursion. That is `h` Kalman steps per period, against `h²` in the Kim filter. With `h = 1` it reduces to the Kalman filter.
- **Smoothing (§4.2):** probabilities by the backward recursion of Kim (1994); states by the smoother of Hashimzade et al. (2024).
- **Estimation (§5.1 and §7.4):** prior per parameter as `{initial value, mean, standard deviation, distribution, lower bound, upper bound}`; mode of the posterior kernel with `fmincon` (default) or `bee_gate`; Hessian at the mode for standard deviations; restart from the previous mode. At least as many shocks as observables are required.
- **MCMC (§5.2):** RWMH with scaled Hessian covariance (the example uses 0.15), several chains, target acceptance of 20% to 40%; the objective function is extracted with `solve_check_stability = false` and `fix_point_TolFun = 1e-6` for speed; MDD by the bridge sampling of Meng and Wong (1996).
- **Counterfactuals (§6):** conditional simulation with a shock plan and a regime path. In models with regimes an exact replica of history is not possible, since the regime is latent; one conditions on the most likely regime according to the smoothed probabilities.
- **Solution in practice (§7.2):** alternative solvers `mn`, `mnk`, `sims`; `solve_order = k` requires derivatives of order k, symbolic (`max_deriv_order`) or by automatic differentiation; the 1st order is always computed first.

### C. Hashimzade, Kirsanov, Kirsanova and Maih (2024, WP 8/2024; published in JBES in 2026)

- Two families (§2.2): GPB(N) keeps `h^N` histories and collapses **after** the state and measurement updates; IMM(N) mixes **after** moving forward in time and **before** the measurement update. Kim-Nelson is GPB(2); the IMM of Blom and Bar-Shalom is IMM(1).
- Result (WP abstract): the canonical IMM is faster and no less accurate, often more so, than its competitors in both families, including Kim-Nelson; using the corresponding smoother improves accuracy in recovering unobserved variables by about 25%.
- Smoothed probabilities (§2.3.1, eq. 14, Algorithm 3): backward recursion with the approximation `Pr[H_t | s_{t+1}, Y_n] ≈ Pr[H_t | s_{t+1}, Y_t]`.
- Smoothed states (§2.3.2, eqs. 15 to 23, Lemma 1): adaptation of the smoother of Durbin and Koopman (2012) and De Jong (1988), which only reuses the `F^{-1}` inversions already done in the filter. The WP describes the state smoother of Kim (1994) as numerically unstable because it inverts large auxiliary matrices, and notes that engineering smoothers require an invertible measurement error covariance, which rarely holds in DSGE models.

### D. Five decisions that follow from the summary

1. Store the structural derivatives per pair `(r, r')` already probability-weighted; `A+_{r,r'}` cannot be collapsed into `A+_r`.
2. A single coupled Sylvester solver (eq. 24) serves the Newton step and all higher orders; offer a Kronecker version (small models) and a matrix-free TFQMR version (large models).
3. Defaults equal to those of RISE: regime-specific steady state, "backward" initialisation, `mfi` as the solver, with `mn`/`mnk` and random multi-start as alternatives.
4. MSS (eq. 30) only when `Q` is constant; with endogenous probabilities, report that there is no criterion and, for information, the MSS evaluated at the steady-state `Q`.
5. IMM as the default filter, Kim/GPB(2) as a validation alternative, Kim's probability smoother and the state smoother of Hashimzade et al.

---

## Suggested changes to the analysis plan (record in docs/VALIDATION.md)

- Solver validation suite: the ten problems of appendix C of Maih (2015), with the acceptance criterion "same number of real and stable solutions as in Table 2" and the coefficients printed in §6.4.
- Cross-validation of the 1st and 2nd orders against `dsgelab` (partitioned perturbation) in the unique steady state cases, where the two methods must coincide.
- Test of the 1st order with endogenous probabilities against the property of Benigno et al. (2025): equality with the exogenous model evaluated at the steady-state `Q`.
- Convention `Q_ij = Pr(s_{t+1} = j | s_t = i)` throughout the package; test the construction of `Υ` against the definition of Costa et al. (2005, ch. 3) with a scalar case of known answer, because the order of the blocks depends on the dating convention.

## Gaps

- **Maih (2015):** I found no version published in a journal; cited as a WP.
- **Hashimzade et al. (2026, JBES):** DOI and metadata confirmed on Crossref; the publisher's page returned 403, so the published abstract was not read. The identification with WP 8/2024 rests on authors, topic and the search result. The title changed. Final volume and pages are not yet available.
- **Maih and Waggoner (2018), "Perturbation methods for DSGE models with time-varying coefficients and transition matrices":** not found on Crossref or in the search. Not included.
- **Lind, "Regime-switching perturbation for non-linear equilibrium models":** I found no record with a DOI. Not included.
- **Juillard and Maih (2010)** (cited by Maih, 2015, for anticipated shocks): not found. Not included.
- **Svensson and Williams, "Distribution forecast targeting":** I only found WP versions (NBER 11733, CEPR 6331). The chapter "Optimal monetary policy under uncertainty in DSGE models: a Markov jump-linear-quadratic approach" (Central Bank of Chile, 2009, vol. 13, pp. 77-114) appears on [RePEc](https://econpapers.repec.org/RePEc:chb:bcchsb:v13c03pp077-114), without a DOI; it did not go into the `.bib`.
- **Meng and Wong (1996, Statistica Sinica), bridge sampling:** I found no DOI. Not in the `.bib`.
- **Gupta, Murray and Hassibi (2003):** Crossref returns an ACC 2003 paper (10.1109/ACC.2003.1243762) with a title different from the expected one; I did not confirm that it is the one cited by Maih. Not in the `.bib`.
- **Leith et al. (2025)** cited in the tutorial: I could not identify it with confidence (Crossref shows Kirsanova, Leith, Machado and Ribeiro, SSRN 2024, which may be the WP version). Not in the `.bib`.
- **Kim and Nelson (1999):** the MIT Press DOI exists, but the Crossref metadata lists the authors incorrectly ("Kim, Halbert"). The `.bib` uses the correct authors.
- **PDFs not read:** Binning and Maih (2015, 2016, 2017), Chang, Maih and Tan (2021), Barthélemy and Marx (2017), Benigno et al. (2025), Albertini and Moyen (2024), Li (2026), Cho (2016, 2021). The summaries in the table come from abstracts, metadata and what Maih (2015) says about them. The sigma-point, OBC-via-regimes and threshold-switching algorithms **still need to be read** before implementing.
- **Criticism of the Kim filter:** apart from Hashimzade et al. and Li (2026), I found in the sources consulted no work from 2021 to 2026 dedicated to quantifying the approximation error of the Kim filter in the MS-DSGE likelihood.
- **Software:** I did not inspect the contents of the supplementary zip of Foerster et al. or of the Zenodo package of Benigno et al.; I did not run `dsgelab`; the status of DSGE.jl comes from a search result; MS-DSGE support in Dynare beyond the `markov_switching` syntax was not clarified.
- **Qualis/CAPES** ranking of the journals: not consulted.

---

## 2026-11-22: optimal policy in linear regime-switching models (commitment, discretion, loose commitment, simple rules)

- **Method:** linear-quadratic optimal policy with stochastic reoptimisation; fixed point in the policy function.
- **Topic:** RISE `planner_objective` block (module `optimal_policy.py`). **Context:** replica of RISE, no data.

Search terms and sources consulted: Crossref (DOI confirmation for every row below), full text of Debortoli, Maih and Nunes in the IFDP 1034 version of the Federal Reserve Board (Nov 2011, read in this session; the equation numbering below is from that version), RISE examples and help texts. The forward-citation search on OpenAlex failed because of the request limit (see Gaps).

| Reference | Type | What it contributes | Implication for the project | Package |
|---|---|---|---|---|
| Debortoli, Maih and Nunes (2014), Macroeconomic Dynamics 18(1), 175-198. [10.1017/S1365100512000326](https://doi.org/10.1017/S1365100512000326) | Published (the IFDP 1034 version was read) | Problem (4), Lagrangian (6), FOCs (9)-(10), envelope condition (11), system (15) and fixed-point algorithm in H (steps 1 to 4); welfare (17)-(22) | **Specification of the module.** `solve_optimal_policy` builds Γ0 and Γ1 of (15) and iterates H = -Γ0⁻¹Γ1; with regimes, the expectations become Q-weighted sums and the `loosecommit` chain | RISE; authors' Dynare code (not run) |
| Schaumburg and Tambalotti (2007), JME 54(2), 302-324. [10.1016/j.jmoneco.2005.08.018](https://doi.org/10.1016/j.jmoneco.2005.08.018) | Published | "Quasi-commitment": reoptimisation with constant probability in the NK model | Origin of the formulation; serves as a test case (canonical NK between discretion and commitment) | n/a |
| Debortoli and Nunes (2010), JET 145(3), 1005-1032. [10.1016/j.jet.2009.10.003](https://doi.org/10.1016/j.jet.2009.10.003) | Published | Loose commitment in fiscal policy; proof that the multiplier is reset to zero on reoptimisation | Justifies zeroing the multiplier columns in the "plan redone" regime | n/a |
| Roberds (1987), "Models of policy under stochastic replanning". Version found: Fed Minneapolis Staff Report 104 (1986). [10.21034/sr.104](https://doi.org/10.21034/sr.104) | WP (the DOI of the International Economic Review version was not confirmed) | First formulation of stochastic replanning | Historical citation; gives its name to the RISE folder `StochasticReplanning` | n/a |
| Debortoli and Lakdawala (2016), AEJ: Macroeconomics 8(3), 42-76. [10.1257/mac.20150029](https://doi.org/10.1257/mac.20150029) | Published | Estimates the Fed's probability of commitment in a Smets-Wouters model, with reoptimisations as a Markov regime | It is the application that `Estimation_examples/usmodel_sr_switch.rs` reproduces; the solution returned here goes into the package's switching filter | n/a |
| Clarida, Galí and Gertler (1999), JEL 37(4), 1661-1707. [10.1257/jel.37.4.1661](https://doi.org/10.1257/jel.37.4.1661) | Published | Closed-form solution of the canonical NK model under discretion and under commitment | Analytical test in `tests/test_optimal_policy.py` | n/a |
| Dennis (2007), Macroeconomic Dynamics 11(1), 31-55. [10.1017/S1365100507050341](https://doi.org/10.1017/S1365100507050341) | Published | Algorithms for commitment and discretion in structural form, without separating predetermined variables | The "Dennis routine" cited in `usmodel_ms_lc.rs`; alternative not implemented | n/a |
| Söderlind (1999), EER 43(4-6), 813-823. [10.1016/S0014-2921(98)00096-8](https://doi.org/10.1016/S0014-2921(98)00096-8) | Published | Classical algorithms (Schur for commitment, iteration for discretion, simple rule) | Reference for the simple-rule loss (`loss`, `optimal_simple_rule`) | n/a |
| Blake and Zampolli (2011), JEDC 35(10), 1626-1651. [10.1016/j.jedc.2011.04.001](https://doi.org/10.1016/j.jedc.2011.04.001) | Published (already in the main table) | Commitment and discretion in MSRE | Reference for commitment and discretion with regimes. Not reread in this session: the dating of `timing="exact"` (lead matrix of the previous regime on the lagged multiplier) is our own derivation and must be checked against the paper | n/a |
| Blake, Dennis, Kirsanova and Yates (2025), Oxford Review of Economic Policy 41(2), 404-425. [10.1093/oxrep/graf022](https://doi.org/10.1093/oxrep/graf022) | Published (abstract and metadata; text not read) | Multiplicity of equilibria without commitment | The iteration returns one discretionary equilibrium, which may not be unique; the package does not enumerate | n/a |
| Blake and Kirsanova, "Discretionary policy and multiple equilibria in LQ RE models". [10.2139/ssrn.943032](https://doi.org/10.2139/ssrn.943032) | WP (SSRN; the DOI of the 2012 REStud version was not confirmed) | Multiple discretionary equilibria in LQ models | Same; cited by Debortoli, Maih and Nunes (footnote 8) | n/a |

### Technical summary (what was implemented)

With `m` private equations `Am_s y_{t-1} + A0_s y_t + Ap_s E_t y_{t+1} + B_s e_t = 0` and objective `u = ½ y'W y`, the planner's FOCs with probability `γ_s` of keeping the plan are (eq. 10 of Debortoli, Maih and Nunes, with `W` here equal to `2W` there):

`W y_t + [A0_s' + (1-γ_s) Σ_s' Q[s,s'] H_yy(s')' Ap_s'] λ_t + β⁻¹ Ap_{s(t-1)}' λ_{t-1}·1{plan kept} + β E_t[Am_{s(t+1)}' λ_{t+1}] = 0`.

The expectation of `λ_{t+1}` mixes continuation (weight `γ_s`) and plan redone (`1-γ_s`, with `λ_t` set to zero), which is equivalent to an MSRE on the `loosecommit` chain × the model's chains; the term in `H_yy` (the effect of `y_t` on what the next planner will do) is what does not fit in an ordinary MSRE solver and requires the fixed point. The iteration is `T_r ← -U_r(T)⁻¹ Γm_r`. The extension to regime-dependent `γ` and coefficients is our own derivation from the Lagrangian (6) with transition probabilities; I found no paper that writes it in this general form.

Welfare: `V_r = W_r/2 + β Σ_r' Q[r,r'] T_r'ᵀ V_r' T_r'`, `d_r = β Σ_r' Q[r,r'] (d_r' + tr(V_r' R_r' Σ_e R_r'ᵀ))`, computed directly from the law of motion. Formulas (17)-(19) of the paper (via multipliers) were not used.

Changes to the analysis plan: none to the roadmap; phase 9 now has a closed specification. Recorded in `docs/VALIDATION.md`.

Gaps:
- OpenAlex returned "rate limit exceeded" in the search for work from 2019 onwards citing Debortoli, Maih and Nunes (2014); the "recent advances" front was restricted to what Crossref returned. To be checked.
- Roberds (1987, International Economic Review) and Blake and Kirsanova (2012, Review of Economic Studies): I did not confirm the DOIs of the journal versions; the `.bib` has the confirmed WP versions.
- Oudiz and Sachs (1985) and Backus and Driffill (1986), classical discretion algorithms: I did not search for DOIs; not in the `.bib`.
- I did not run the authors' Dynare code (`fm95_lc.dyn` uses routines that do not ship with standard Dynare); the comparison is with RISE only.

## 2026-11-22: occasionally binding constraints (module `occbin.py`)

**Method:** occasionally binding constraints in 1st-order models treated as regimes (RISE `?` lines and `solve_occbin`).
**Sources consulted:** PDF of Norges Bank WP 23/2017 downloaded from the bank's page and read (92 pp., text extracted with `pdftotext`); Crossref (`/works/<DOI>`) for the two DOIs below; open source code of the OccBin that ships with Dynare 6.4 (`matlab/+occbin/solve_one_constraint.m`, `setup.m`); RISE help texts and examples (`examples/occbin`, `m/+utils/+forecast/*.m`).

| Reference (DOI/link) | Type | What it contributes | Implication for the project |
|---|---|---|---|
| Binning and Maih (2017), "Modelling occasionally binding constraints using regime-switching", Norges Bank WP 23/2017. [page](https://www.norges-bank.no/en/news-events/publications/Working-Papers/2017/232017/), SSRN DOI [10.2139/ssrn.3073753](https://doi.org/10.2139/ssrn.3073753) (title and authors checked on Crossref) | WP (no journal version found) | Section 2: the constraint becomes a two-state chain (N, B) and an indicator parameter φ; complementary slackness (2.2) becomes `φ(A−B) + (1−φ)λ = 0` (eq. 2.4) and the max/min (2.3) becomes `C = φD + (1−φ)F` (eq. 2.5). The probabilities are endogenous and logistic (eqs. 2.7, 2.8, 4.1), with θ = 1 and ψ = 2000 in the ZLB example (eqs. 5.6, 5.7). Regime-specific steady state at the ZLB (eq. 5.8). Section 8: comparison with the piecewise-linear solution and with extended path | It is the formulation of the equations used in the examples (`R = bind*r_zlb + (1-bind)*R_TAYLOR`). **The paper does not describe the algorithm of `solve_occbin`/`simul_honor_constraints`**: the piecewise-linear solution appears only as a comparison method (Guerrieri and Iacoviello). The paper's version (endogenous probabilities, `!` lines) is already covered by the package's solver. In section 8.1 the authors record that, with `p_NZ = 0` and `p_ZN = 1`, the IRFs of the normal regime coincide with those of the piecewise-linear solution |
| Guerrieri and Iacoviello (2015), JME 70, 22-38. [10.1016/j.jmoneco.2014.08.005](https://doi.org/10.1016/j.jmoneco.2014.08.005) (checked on Crossref) | Published | Piecewise-linear solution: guess of the regime sequence, time-varying decision rules computed backwards from the reference solution, verification and update of the guess | Algorithm implemented in `occbin.py`. RISE (2019 version) differs from the original in two points, identified by numerical comparison; see `docs/VALIDATION.md` |

**Gaps.** There is no documentation of RISE's simulation algorithm (`simul_occbin`, `one_step_fbs`, `simul_forward_back_shooting` have empty help); the update rule for the regime sequence was inferred from the results. Holden and Paetz (2012) and Juillard and Maih (2010), cited by Binning and Maih for imposing constraints through anticipated shocks, were neither read nor verified and stay out of the `.bib`.

## 2026-11-23: Bayesian estimation (samplers, marginal data density, diagnostics)

References for the algorithms in `samplers.py`, `mdd.py`, `optimizers.py` and `diagnostics.py`. DOIs checked
on Crossref (query by title and authors); entries in `references.bib`.

| Use in the package | Reference | DOI |
|---|---|---|
| Adaptive scale and covariance of the MH; APT | Miasojedow, Moulines and Vihola (2013), JCGS 22(3), 649-664 | 10.1080/10618600.2013.778779 |
| DSMH | Waggoner, Wu and Zha (2016), J. Econometrics 192(2), 406-420 | 10.1016/j.jeconom.2016.02.007 |
| SMC | Herbst and Schorfheide (2014), JAE 29(7), 1073-1098 | 10.1002/jae.2397 |
| `laplace` | Tierney and Kadane (1986), JASA 81(393), 82-86 | 10.1080/01621459.1986.10478240 |
| `laplace_mcmc` | Lewis and Raftery (1997), JASA 92(438), 648-655 | 10.1080/01621459.1997.10474016 |
| `mhm` | Gelfand and Dey (1994), JRSS B 56(3), 501-514; Geweke (1999), Econometric Reviews 18(1), 1-73 | 10.1111/j.2517-6161.1994.tb01996.x; 10.1080/07474939908800428 |
| `swz` | Sims, Waggoner and Zha (2008), J. Econometrics 146(2), 255-274 | 10.1016/j.jeconom.2008.08.023 |
| `mueller` | Liu, Waggoner and Zha (2011), QE 2(2), 251-301 (description of the method; Mueller's text was not published and I could not locate it) | 10.3982/QE71 |
| `bridge` | Meng and Wong (1996), Statistica Sinica 6, 831-860 | no DOI on Crossref |
| `is`, `ris` | Frühwirth-Schnatter (2004), Econometrics Journal 7(1), 143-167 | 10.1111/j.1368-423X.2004.00125.x |
| `cj` | Chib and Jeliazkov (2001), JASA 96(453), 270-281 | 10.1198/016214501750332848 |
| `ce` (not in RISE) | Chan and Eisenstat (2015), Econometric Reviews 34(3), 256-285 | 10.1080/07474938.2014.944474 |
| `bee_gate` | Karaboga and Basturk (2007), J. Global Optimization 39(3), 459-471 | 10.1007/s10898-007-9149-x |
| PSRF | Gelman and Rubin (1992), Statistical Science 7(4), 457-472; Brooks and Gelman (1998), JCGS 7(4), 434-455 | 10.1214/ss/1177011136; 10.1080/10618600.1998.10474787 |
| Effective sample size | Geyer (1992), Statistical Science 7(4), 473-483; Vehtari et al. (2021), Bayesian Analysis 16(2), 667-718 | 10.1214/ss/1177011137; 10.1214/20-BA1221 |
| Geweke | Geweke (1992), Bayesian Statistics 4, 169-193 | 10.1093/oso/9780198522669.003.0010 |
| Local identification | Iskrev (2010), JME 57(2), 189-202 | 10.1016/j.jmoneco.2009.12.007 |
| Endogenous priors | Christiano, Trabandt and Walentin (2011), JEDC 35(12), 1999-2041; Del Negro and Schorfheide (2008), NBER WP 13741 | 10.1016/j.jedc.2011.09.005; 10.3386/w13741 |

Not verified by reading the full text in this round: the formulas of `swz`, `mueller`, DSMH and APT were
written from knowledge of the articles and checked by simulation on targets with a known constant
(`tests/test_mdd.py`, `tests/test_samplers.py`), not equation by equation against the PDFs.

## Addendum (2026-11-23): Binning and Maih (2015), read

PDF read: [Norges Bank WP 10/2015](https://www.norges-bank.no/contentassets/db9d942135b449c2b444f6e78750353d/working_paper_10_15.pdf) (version of 18 May 2015; accessed on 2026-11-22). Summary of what was used in `nonlinear_filters.py`:

- **Algorithm 2 (§2.3), generic sigma-point filter with regimes:** per regime, forecast of the observable, marginal likelihood and Kalman update; update of the probabilities; `Q` evaluated at the updated state of "any" regime; collapse **before** the forecast, `x̂(r') = Σ_r p(r,r') x_t|t(r)` and `P̂(r') = Σ_r p(r,r') P_t|t(r)`, with `p(r,r') = Q(r,r') p_t|t(r) / p_t+1|t(r')`, without the spread-of-means term; sigma points over `X = [x_{t-1}, η_t]`.
- **§2.3.1, rules:** UKF with `n_X + λ = 3` (weights `λ/3` and `1/6`); DDF with Stirling interpolation, `δ = √3`, mean weights `(δ² − n_X)/δ²` and `1/(2δ²)`, factors `S1 = (f(X_j) − f(X_{n+j}))/(2δ)` and `S2 = √(δ² − 1)/(2δ²) (f(X_j) + f(X_{n+j}) − 2 f(X_0))`, covariance by triangularisation of `[S1, S2]`; CKF with `2 n_X` points at `±√n_X` and equal weights.
- **Covariance correction:** a matrix that is not positive definite is replaced by the nearest positive definite one. The paper does not give the rule; in RISE it is a `sqrt(eps)` floor on the eigenvalues (identified numerically).
- The paper acknowledges that the collapse violates Jensen's inequality and defends it by simulation.
