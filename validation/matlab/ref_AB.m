rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
out=struct(); names={'A','B'};
for k=1:2
  r=struct();
  try
    m=rise(['ref' names{k}]); [m,retcode]=solve(m); r.retcode=retcode;
    sol=m.solution; r.Tz=cellfun(@full,sol.Tz,'uni',0); r.ss=cellfun(@full,sol.ss,'uni',0);
    r.Q=full(sol.transition_matrices.Q); r.endo={m.endogenous.name}; r.order_var=m.order_var;
    r.stable=is_stable_system(m);
    y=readmatrix(['../ref/ref' names{k} '_data.csv']); db=struct('X',ts('1990Q1',y));
    [f,LogLik,Incr,rc]=filter(m,'data',db,'kf_riccati_tol',0);
    r.LogLik=LogLik; r.Incr=Incr; r.frc=rc;
    r.upd1=double(f.updated_regime_probabilities.regime_1); r.smo1=double(f.smoothed_regime_probabilities.regime_1);
    r.updZ=double(f.Expected_updated_variables.Z); r.smoZ=double(f.Expected_smoothed_variables.Z);
    r.smoE=double(f.Expected_smoothed_shocks.E);
    myirf=irf(m,'irf_periods',10); r.irf_fields=fieldnames(myirf.E); r.irfZ=double(myirf.E.Z); r.irfX=double(myirf.E.X);
  catch err, r.error=getReport(err); end
  out.(names{k})=r;
end
save('../ref/refAB.mat','-struct','out','-v7'); disp('OK');
