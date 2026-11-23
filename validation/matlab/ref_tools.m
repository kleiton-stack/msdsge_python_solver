% RISE references for forecasting.py, analysis.py (remaining items), display.py and timeseries.py
rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
here=pwd; work=tempname; mkdir(work);
fl={'tool_c.rs','tool_same.rs','flt_ms.rs','refA.rs'}; for k=1:numel(fl), copyfile(fl{k},work); end
cd(work);
out=struct();
y=readmatrix([here '/../ref/filter_ms_data.csv']); db=struct('x',ts('1990Q1',y));
names={'tool_c','flt_ms'};
for k=1:2
  nm=names{k}; r=struct();
  m=rise(nm); m=solve(m); m=set(m,'data',db,'kf_riccati_tol',0);
  try
    hd=historical_decomposition(m); r.hd_x=double(hd.x); r.hd_z=double(hd.z); r.hd_names=hd.x.varnames;
  catch err, r.hd_error=getReport(err); disp(getReport(err)); end
  try
    [ac,rc]=theoretical_autocovariances(m,'autocov_ar',3); r.ac=ac;
    r.acorr=theoretical_autocorrelations(m,'autocov_ar',3);
  catch err, r.ac_error=getReport(err); disp(getReport(err)); end
  try
    r.print=evalc('print_solution(m)'); r.print_x=evalc('print_solution(m,''x'')'); r.resid=evalc('resid(m)'); r.resid_val=resid(m);
  catch err, r.print_error=getReport(err); disp(getReport(err)); end
  try
    vd=variance_decomposition(m,'vardec_ergodic',true);
    r.vde_cond_z=double(vd(1).conditional.z); r.vde_cond_x=double(vd(1).conditional.x);
    r.vde_inf_z=double(vd(1).infinity.z); r.vde_inf_x=double(vd(1).infinity.x);
  catch err, r.vde_error=getReport(err); disp(getReport(err)); end
  % forecasts: database with smoothed variables and shocks
  try
    [f,LL]=filter(m);
    sv=f.Expected_smoothed_variables; ss=f.Expected_smoothed_shocks;
    full=struct('x',sv.x,'z',sv.z,'e',ss.e,'u',ss.u);
    r.smo=[double(sv.x),double(sv.z),double(ss.e),double(ss.u)];
    if k==1, reg=ones(1,8); else, reg=[1 1 2 2 1 1 1 1]; end
    % (a) no uncertainty, imposed regimes, history up to 2100Q1 (obs 441)
    fk=forecast(m,'data',full,'forecast_start_date','2100Q2','forecast_nsteps',8,'simul_regime',reg);
    r.fa_fields=fieldnames(fk); r.fa=[double(fk.x),double(fk.z),double(fk.e),double(fk.u),double(fk.regime)]; r.fa_start=fk.x.start;
    % (b) drawn regimes
    rng(3); fk=forecast(m,'data',full,'forecast_start_date','2100Q2','forecast_nsteps',8);
    r.fb=[double(fk.x),double(fk.z),double(fk.e),double(fk.u),double(fk.regime)];
    % (c) with shock uncertainty
    rng(4); fk=forecast(m,'data',full,'forecast_start_date','2100Q2','forecast_nsteps',8,'forecast_shock_uncertainty',true);
    r.fc=[double(fk.x),double(fk.z),double(fk.e),double(fk.u),double(fk.regime)];
    % (d) conditional on shock e (Ireland): pages with the future
    pg=pageify(date2serial('2100Q1'),full); r.pg_class=class(pg); r.pg_size=size(double(pg)); r.pg_names=pg.varnames; r.pg_start=pg.start; r.pg_data=double(pg);
    fk=forecast(m,'data',pg,'forecast_start_date','2100Q2','forecast_nsteps',8,'forecast_cond_exo_vars','e','simul_regime',reg);
    r.fd=[double(fk.x),double(fk.z),double(fk.e),double(fk.u),double(fk.regime)];
    % (e) conditional on endogenous x
    fk=forecast(m,'data',pg,'forecast_start_date','2100Q2','forecast_nsteps',8,'forecast_cond_endo_vars','x','simul_regime',reg);
    r.fe=[double(fk.x),double(fk.z),double(fk.e),double(fk.u),double(fk.regime)];
    % (f) conditional on x and on u
    fk=forecast(m,'data',pg,'forecast_start_date','2100Q2','forecast_nsteps',8,'forecast_cond_endo_vars','x','forecast_cond_exo_vars','u','simul_regime',reg);
    r.ff=[double(fk.x),double(fk.z),double(fk.e),double(fk.u),double(fk.regime)];
    hyp={'jma','ncp','nas'};
    for q=1:3
      try
      fk=forecast(m,'data',pg,'forecast_start_date','2100Q2','forecast_nsteps',8,'forecast_cond_endo_vars','x','simul_regime',reg,'forecast_conditional_hypothesis',hyp{q});
      r.(['fe_' hyp{q}])=[double(fk.x),double(fk.z),double(fk.e),double(fk.u),double(fk.regime)];
      catch err, r.(['fe_' hyp{q} '_error'])=getReport(err); end
    end
    % (g) conditional on x with uncertainty
    rng(5); fk=forecast(m,'data',pg,'forecast_start_date','2100Q2','forecast_nsteps',8,'forecast_cond_endo_vars','x','simul_regime',reg,'forecast_shock_uncertainty',true);
    r.fg=[double(fk.x),double(fk.z),double(fk.e),double(fk.u),double(fk.regime)];
    % (h) condition on x in the first 3 steps only
  catch err, r.fk_error=getReport(err); disp(getReport(err)); end
  try
    if k==1, sr=ones(1,20); else, sr=[1 1 1 2 2 2 1 1 1 1 2 2 1 1 1 1 1 1 1 1]; end
    rng(6); [sim,states]=simulate(m,'simul_periods',20,'simul_burn',0,'simul_regime',sr);
    r.sim_fields=fieldnames(sim); r.sim=[double(sim.x),double(sim.z),double(sim.e),double(sim.u)]; r.sim_states=states; r.sim_start=sim.x.start;
    rng(7); [sim,states]=simulate(m,'simul_periods',20,'simul_burn',0);
    r.sim2=[double(sim.x),double(sim.z),double(sim.e),double(sim.u)]; r.sim2_states=states;
  catch err, r.sim_error=getReport(err); disp(getReport(err)); end
  % counterfactual(m,[],1,{'e'}) fails in RISE itself (commit 1b2edfa, R2024a): 'Brace indexing is not supported' in utils.filtering.nonlinear_shock_decomp
  out.(nm)=r;
end
% refA: regime-specific steady state
try
  yA=readmatrix([here '/../ref/refA_data.csv']); dbA=struct('X',ts('1990Q1',yA));
  m=rise('refA'); m=solve(m); m=set(m,'data',dbA,'kf_riccati_tol',0); r=struct();
  r.print=evalc('print_solution(m)'); r.resid=evalc('resid(m)'); r.resid_val=resid(m);
  r.ac=theoretical_autocovariances(m,'autocov_ar',2);
  f=filter(m); sv=f.Expected_smoothed_variables; ss=f.Expected_smoothed_shocks;
  full=struct('X',sv.X,'Z',sv.Z,'E',ss.E); r.smo=[double(sv.X),double(sv.Z),double(ss.E)];
  r.n=numel(yA);
  reg=[1 2 2 2 1 1];
  fk=forecast(m,'data',full,'forecast_start_date','2000Q1','forecast_nsteps',6,'simul_regime',reg);
  r.fa=[double(fk.X),double(fk.Z),double(fk.E),double(fk.regime)]; r.fa_start=fk.X.start;
  pg=pageify(date2serial('1999Q4'),full);
  fk=forecast(m,'data',pg,'forecast_start_date','2000Q1','forecast_nsteps',6,'forecast_cond_endo_vars','Z','simul_regime',reg);
  r.fe=[double(fk.X),double(fk.Z),double(fk.E),double(fk.regime)];
  vd=variance_decomposition(m,'vardec_ergodic',true); r.vde_cond_Z=double(vd(1).conditional.Z);
  out.refA=r;
catch err, out.refA_error=getReport(err); disp(getReport(err)); end
% time series
try
  rng(9); v=cumsum(randn(48,1))+10; t=ts('1990Q1',v); out.ts_v=v;
  [tr,cy]=hpfilter(t,1600); out.ts_hp_trend=double(tr); out.ts_hp_cycle=double(cy);
  a=aggregate(t,'H'); out.ts_agg_H=double(a); out.ts_agg_H_start=a.start;
  vm=cumsum(randn(60,1))+20; tm=ts('1990M1',vm); out.ts_vm=vm;
  a=aggregate(tm,'Q'); out.ts_agg_Q=double(a); out.ts_agg_Q_start=a.start;
  a=aggregate(tm,'Q','interpolation'); out.ts_agg_Qi=double(a);
  [tr,dt]=ma_filter(t,2); out.ts_ma_trend=double(tr); out.ts_ma_start=tr.start;
  [tr,dt]=ma_filter(t,2,true); out.ts_ma_trend_ext=double(tr);
  [tr,dt]=ma_filter(t); out.ts_ma_default=double(tr);
  out.ts_serial=date2serial('1990Q1'); out.ts_serial_m=date2serial('1990M3'); out.ts_dates=serial2date(date2serial('1990Q1'):date2serial('1990Q4'));
  c=cumsum(t); out.ts_cumsum=double(c);
  out.ts_log=double(log(t)); x2=t('1991Q2:1992Q1'); out.ts_window=double(x2); out.ts_window_start=x2.start;
  tt={'pct_ch_ar','pct_ch','yoy_pct_ch','diff','yoy_diff','log_ch_ar','log_ch','yoy_log_ch'};
  % ts/transform fails in RISE (commit 1b2edfa, R2024a) both with a numeric code and with a name
  for q=1:numel(tt), a=transform(t,q+1); out.(['ts_tr_' tt{q}])=double(a); out.(['ts_tr_' tt{q} '_start'])=a.start; end
catch err, out.ts_error=getReport(err); disp(getReport(err)); end
cd(here); save('../ref/tools.mat','-struct','out','-v7'); disp('OK');
