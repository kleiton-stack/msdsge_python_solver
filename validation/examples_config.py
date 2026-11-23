"""Flags, parameters and options that the MATLAB scripts of the RISE examples pass to the models.

``CONFIG[file]`` is a list of variants (one per relevant call in the driver). Fields:

- ``key``: name of the case in the RISE reference (``validation/ref/examples.mat``).
- ``flags``: macro-language variables (``rise_flags`` in the driver).
- ``driver``: MATLAB script from which the flags, parameters and options were read.
- ``matlab``: code that builds the model ``m`` in RISE as the driver does and, when the driver sets
  parameters, leaves them in ``p_in``. ``validation/gen_ref_examples.py`` uses this code to run
  RISE; the parameters in ``p_in`` come back in ``examples.mat`` and are read by ``parameters_of``.
- ``msolve``: ``solve`` options in RISE; ``solve``: the equivalent options in this package.
- ``ss``: ``"py:name"`` uses the Python function in ``examples_ssfiles.py`` (hand port of the
  ``.m`` file); ``"tab"`` uses the steady state and the new parameters that the example's ``.m``
  file returned in RISE, tabulated in ``examples.mat`` (valid only at the driver's parameters).

``FRAGMENTS``: files that exist only to be included by others (``@#include``) or that the
examples do not calibrate anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REF = Path(__file__).parent / "ref" / "examples.mat"

_TZ = "VariousModels/TaoZha/Tutorials/DSGE/"
_PI = "VariousModels/PeterIreland/"
_FR = "MarkovSwitching/FoersterRubioRamirezWaggonerZha/"

FRAGMENTS = {
    **{_TZ + f"Tutorial1/param_{k}.rs": "included by Tutorial1/fwz.rs"
       for k in ("original", "policyChangeOnly", "privateChangeOnly", "volatilityChangeOnly")},
    _TZ + "Tutorial1/switching_policy_parameters.rs": "parameter fragment; no file includes it",
    **{_TZ + f"Tutorial2/Models/{k}.rs": "included by the Tutorial2 models"
       for k in ("constant_policy_parameters", "constant_volatility", "fwz_est", "switching_volatility",
                 "switching_policy_parameters_coefChain", "switching_policy_parameters_volChain")},
    **{_TZ + f"Tutorial3/Models/{k}.rs": "included by the Tutorial3 models"
       for k in ("constant_policy", "constant_volatility", "svar_core", "switching_policy", "switching_volatility")},
    _FR + "wp1301/rbcns.rs": "no calibration: no example driver uses the file",
}

# part of the FarmerWaggonerZha2010 howto.m that defines the three parameterisations
_FWZ = ("m=rise('fwz10_nasty'); txt=fileread('howto.m'); i=strfind(txt,'pp=struct()'); j=strfind(txt,'np=numel(pp)'); "
        "eval(txt(i(1):j(1)-1)); p_in=pp(%d);")
_STATA = "m=rise('model%s'); p_in=stata_params('driver%s.m');"


def _c(key, matlab, driver, **kw):
    return {"key": key, "matlab": matlab, "driver": driver, **kw}


CONFIG: dict[str, list[dict]] = {
    "MarkovSwitching/FarmerWaggonerZha2010/fwz10_nasty.rs": [
        _c(f"fwz10_{k}", _FWZ % k, "howto.m", label=f"parameterisation {k}", msolve="'solver','mnk'", solve={"solver": "mnk"})
        for k in (1, 2, 3)],
    _FR + "frwz_nk.rs": [_c("frwz_nk", "m=rise('frwz_nk');", "howto.m")],
    _FR + "wp1301/nk.rs": [_c("wp1301_nk", "m=rise('nk');", "driver_nk.m")],
    _FR + "wp1301/nkh.rs": [_c("wp1301_nkh", "m=rise('nkh');", "driver_nkh.m")],
    _FR + "wp1301/rbcs.rs": [_c("wp1301_rbcs", "m=rise('rbcs');", "driver_rbcs.m")],
    _FR + "wp1301/rbcs0.rs": [_c("wp1301_rbcs0", "m=rise('rbcs0');", "driver_rbcs_multipe_solutions.m")],
    _FR + "wp1416/simple.rs": [_c("wp1416_simple", "m=rise('simple');", "driver_simple.m")],
    # the steady state of regime 2 does not exist; the drivers use the FRWZ perturbation, which starts
    # from the ergodic point. The equivalent here is the unique point (steady_state="unique").
    _FR + "wp1416/rbcs.rs": [_c("wp1416_rbcs", "m=rise('rbcs');", "driver_rbcs.m", label="unique point",
                                msolve="'steady_state_unique',true", solve={"steady_state": "unique"})],
    _FR + "QE2016/rbcs.rs": [_c("qe2016_rbcs", "m=rise('rbcs');", "driver_rbcs.m", label="unique point",
                                msolve="'steady_state_unique',true", solve={"steady_state": "unique"})],
    "MarkovSwitching/GavinKeenRichterThrockmorton/globdynzlb.rs": [
        _c("globdynzlb", "m=rise('globdynzlb','rise_flags',struct('exogenous_switching',true));", "howto.m",
           flags={"exogenous_switching": True}, msolve="'steady_state_imposed',true", solve={"steady_state_imposed": True})],
    "MarkovSwitching/GavinKeenRichterThrockmorton/constant.rs": [_c("gkrt_constant", "m=rise('constant');", "(no driver)")],
    "MarkovSwitching/SmetsWouters/usmodel.rs": [
        _c("sw_ms", "m=rise('usmodel','solve_linear',true);", "howto.m", msolve="'solver','mn'", solve={"solver": "mnk"})],
    "VariousModels/SmetsWouters/usmodel.rs": [_c("sw_const", "m=rise('usmodel');", "howto.m")],
    "ModelWithMeasurementErrors/Canonical_Const.rs": [_c("canon_me", "m=rise('Canonical_Const');", "howto.m")],
    "ModelsWithSteadyStateFile/Canonical_Const.rs": [
        _c("canon_ssf", "m=rise('Canonical_Const','steady_state_file','steady_state_4_Canonical_Const');", "howto.m",
           ss="py:canonical_const")],
    "MonteCarloFiltering/trinity.rs": [_c("trinity", "m=rise('trinity');", "howto.m")],
    "Nonlinear3rdOrder/Binning/small_nk.rs": [
        _c("small_nk", "m=rise('small_nk');", "(no driver)"),
        # not from a driver: case added to validate the log approximation
        _c("small_nk_log", "m=rise('small_nk');", "(no driver)", label="log_vars C R K",
           msolve="'solve_log_approx_vars',{'C','R','K'}", solve={"log_vars": ["C", "R", "K"]})],
    "Nonlinear3rdOrder/Binning/rudebusch_swanson.rs": [_c("rudebusch_swanson", "m=rise('rudebusch_swanson');", "(no driver)")],
    "NonlinearModels/fs2000.rs": [
        _c("fs2000_ssfile", "m=rise('fs2000');", "howto.m", label="steady-state file",
           msolve="'steady_state_file','fs2000_steadystate'", ss="py:fs2000")],
    "NonlinearModels/fs2000_b.rs": [
        _c(f"fs2000_b_{a}", f"m=rise('fs2000_b','rise_flags',struct('approximate',{a}));", "howto.m",
           label=f"approximate={a}", flags={"approximate": a}) for a in (0, 1)],
    "VariousModels/FrankSchorfheide/LossFunction_JAE2000/fs2000_rise.rs": [_c("fs2000_rise", "m=rise('fs2000_rise');", "howto.m")],
    "VariousModels/HybridExpectations/hybrid_expect.rs": [_c("hybrid", "m=rise('hybrid_expect');", "howto.m")],
    "VariousModels/KaijiTutorial/kaiji1207_exp.rs": [_c("kaiji", "m=rise('kaiji1207_exp');", "howto.m")],
    "steady_state_operator/tshocksnk.rs": [_c("tshocksnk", "m=rise('tshocksnk');", "(no driver)")],
    "VariousModels/SeonghoonCho/woodford2003.rs": [
        _c(f"woodford_{int(f)}", f"m=rise('woodford2003','rise_flags',struct('multiple',{str(f).lower()}));", "(no driver)",
           label=f"multiple={f}", flags={"multiple": f}) for f in (False, True)],
    # Peter Ireland
    _PI + "Endogenous_JME2003/emosp.rs": [
        _c("emosp", "m=rise('emosp','steady_state_file','sstate_file','steady_state_imposed',true); "
           "p_in=create_parameters(true);", "master.m", ss="tab", solve={"steady_state_imposed": True}),
        # master.m estimates with solve_log_approx_vars = state variables (listed in the driver comment)
        _c("emosp_log", "m=rise('emosp','steady_state_file','sstate_file','steady_state_imposed',true); "
           "p_in=create_parameters(true);", "master.m", label="log_vars of the states", ss="tab",
           msolve="'solve_log_approx_vars',{'A','E','K','M','V','X','Z'}",
           solve={"steady_state_imposed": True, "log_vars": ["A", "E", "K", "M", "V", "X", "Z"]})],
    _PI + "Endogenous_JME2003/emosp_ssmodel.rs": [
        _c("emosp_ssmodel", "m=rise('emosp_ssmodel'); p_in=create_parameters(true);", "master.m (parameters)")],
    _PI + "Endogenous_JME2003/emosp_full.rs": [_c("emosp_full", "m=rise('emosp_full');", "(no driver)")],
    _PI + "Endogenous_JME2003/emosp_linear.rs": [
        _c(f"emosp_linear_{int(o)}", f"m=rise('emosp_linear','rise_flags',{{'original',{str(o).lower()}}},'solve_linear',true); "
           "p_in=create_parameters(true);", "master.m", label=f"original={o}", flags={"original": o}) for o in (False, True)],
    _PI + "Endogenous_JME2003/emosp_ns.rs": [
        _c("emosp_ns", "m=rise('emosp_ns'); p_in=create_parameters_nonstationary(true);", "(no driver; parameters from "
           "create_parameters_nonstationary.m)")],
    _PI + "Method_JEDC2004/ireland2004.rs": [_c("ireland2004", "m=rise('ireland2004');", "driver.m")],
    _PI + "NKperspective_JMCB2011/nkp.rs": [_c("nkp", "m=rise('nkp'); p_in=create_parameters(false);", "master.m")],
    _PI + "NKperspective_JMCB2011/nkp_switch.rs": [
        _c("nkp_switch", "m=rise('nkp_switch'); txt=fileread('master_switch.m'); i=strfind(txt,'p=struct()'); "
           "j=strfind(txt,'m=set(m,''parameters'',p)'); eval(txt(i(1):j(1)-1)); p_in=p;", "master_switch.m")],
    "endogenousPriors/NKperspective_JMCB2011/nkp.rs": [_c("nkp_endo", "m=rise('nkp'); p_in=create_parameters(false);", "master.m")],
    _PI + "Targets_JMCB2007/targets.rs": [
        _c("targets", "m=rise('targets','steady_state_file','sstate_model'); p_in=create_parameters('a',false,false);",
           "master.m", ss="tab")],
    _PI + "Targets_JMCB2007/targets_lin.rs": [
        _c("targets_lin", "m=rise('targets_lin'); p_in=create_parameters('a',true,false);", "master.m")],
    _PI + "productivity_RED2008/pdk.rs": [
        _c("pdk", "m=rise('pdk','rise_flags',{'Sektors',{'C','I'}},'steady_state_file','sstate_model'); "
           "p_in=create_parameters(false);", "master.m", flags={"Sektors": ["C", "I"]}, ss="tab")],
    _PI + "sgusea_JEEA2013/sgusea1.rs": [
        _c("sgusea1", "m=rise('sgusea1','rise_flags',struct('CountryNames',{{'H','F'}})); "
           "m=set(m,'steady_state_file','sstate_model'); p_in=create_parameters(true);", "master.m",
           flags={"CountryNames": ["H", "F"]}, ss="tab")],
    _PI + "sgusea_JEEA2013/sgusea12.rs": [
        _c("sgusea12", "m=rise('sgusea12','rise_flags',struct('CountryNames',{{'H','F'}},'ncp_shocks',true)); "
           "m=set(m,'steady_state_file','sstate_model12b'); p_in=create_parameters12(true);", "master12.m",
           flags={"CountryNames": ["H", "F"], "ncp_shocks": True}, ss="tab")],
    # Liu, Wang and Zha
    "VariousModels/TaoZha/LWZ_Econometrica2013/lwz_model.rs": [
        _c("lwz", "m=rise('lwz_model','solve_linear',true); p_in=create_parameters(true);", "master.m")],
    "VariousModels/TaoZha/LWZ_Econometrica2013/lwz_switch_model.rs": [
        _c("lwz_switch", "m=rise('lwz_switch_model','solve_linear',true); p_in=create_parameters_switch(true);", "master_switch.m")],
    "VariousModels/frb/EDO/edo.rs": [
        _c("edo", "m=rise('edo'); p_in=edo_params();", "driver.m", msolve="'steady_state_file','ssfile'", ss="tab")],
    # OccBin: only the reference regime here (the piecewise solution belongs to msdsge.occbin)
    "occbin/oneConstraint/occbin.rz": [
        _c("occbin_one", "m=rise('occbin'); p_in=baselineParameters(); p_in.r_zlb=1; p_in.bind_ocb_1=0; p_in.bind_ocb_2=1; "
           "p_in.ocb_tp_1_2=0; p_in.ocb_tp_2_1=0; m=set(m,'steady_state_file','ssfile_full');", "driver.m", ss="tab",
           msolve="'solve_occbin',1,'steady_state_imposed',true,'steady_state_unique',true",
           solve={"steady_state_imposed": True})],
    "occbin/multipleConstraints/ig2017.rs": [
        _c("occbin_ig2017", "m=rise('ig2017'); p_in=paramfile(); m=set(m,'steady_state_file','ssfile',"
           "'solve_occbin',{1,struct('debt',1,'zlb',2),m.markov_chains.regimes});", "driver.m", ss="tab",
           msolve="'steady_state_unique',true,'steady_state_imposed',true", solve={"steady_state_imposed": True})],
    # Tao Zha tutorials
    _TZ + "Tutorial1/fwz.rs": [
        _c(f"tz1_fwz_{k}", f"m=rise('fwz','solve_linear',true,'rise_flags',struct('indx_model',{k}));", "howto.m",
           label=f"indx_model={k}", flags={"indx_model": k}) for k in (0, 1, 2, 3)],
    **{_TZ + f"Tutorial2/Models/{k}.rs": [_c(f"tz2_{k}", f"m=rise('{k}','solve_linear',true);", "howto.m")]
       for k in ("policyOnly", "volatilityOnly", "volatilityPolicyIndependent", "volatilityPolicyIndependent_2",
                 "volatilityPolicySame")},
    **{_TZ + f"Tutorial3/Models/{k}.rs": [_c(f"tz3_{k}", f"m=rise('{k}','solve_linear',true);", "howto.m")]
       for k in ("svar_constant", "svar_policy", "svar_policy_volatility", "svar_volatility")},
    # models from the Stata manual: the drivers only give starting values (priors) and a few fixed parameters
    **{f"VariousModels/stata/model{k}.rs": [_c(f"stata{k}", _STATA % (k, k), f"driver{k}.m")]
       for k in ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12")},
}

# example folders (the MATLAB code runs in a temporary copy of the folder)
FOLDER = {v["key"]: str(Path(rel).parent) for rel, vs in CONFIG.items() for v in vs}


def _ref():
    import scipy.io as sio
    if not REF.exists():
        return {}
    return sio.loadmat(REF, squeeze_me=True, struct_as_record=False)


_CACHE: dict = {}


def reference(key: str):
    """RISE result for the case ``key`` (or None if the reference has not been generated)."""
    if "ref" not in _CACHE:
        _CACHE["ref"] = _ref()
    return _CACHE["ref"].get(key)


def parameters_of(key: str) -> dict | None:
    """Parameters set by the driver (``p_in``), as {name: value}."""
    r = reference(key)
    if r is None or not getattr(r, "p_in_json", None):
        return None
    out = {}
    for k, v in json.loads(str(r.p_in_json)).items():
        if v is not None and np.isfinite(v):
            out[k] = float(v)
    return out


def tabulated_steady_state(key: str):
    """Steady-state function with the values that the example's ``.m`` file returned in RISE.

    Valid only at the driver's parameters: good for the sweep, not a substitute for porting the file.
    """
    r = reference(key)
    if r is None or getattr(r, "retcode", 1) != 0:
        return None
    ss = dict(zip([str(x) for x in np.atleast_1d(r.endo)], np.atleast_2d(np.asarray(r.ss, float).T).T.reshape(len(np.atleast_1d(r.endo)), -1)[:, 0]))
    par = dict(zip([str(x) for x in np.atleast_1d(r.par_names)], np.asarray(r.par_values, float).reshape(len(np.atleast_1d(r.par_names)), -1)[:, 0]))

    def f(model, p):
        out = {k: v for k, v in ss.items() if k in model.endogenous}
        out.update({k: par[k] for k in model.parameters if k in par and not np.isfinite(p[k]) and np.isfinite(par[k])})
        return out
    f.imposed = True
    return f
