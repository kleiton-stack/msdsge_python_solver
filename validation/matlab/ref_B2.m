rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
m=rise('refB'); m=solve(m);
y=readmatrix('../ref/refB_data.csv'); db=struct('X',ts('1990Q1',y));
[f,LogLik,Incr]=filter(m,'data',db,'kf_riccati_tol',0);
out=struct('LogLik',LogLik,'Incr',Incr);
out.filt1=double(f.filtered_regime_probabilities.regime_1); out.upd1=double(f.updated_regime_probabilities.regime_1);
out.updZ=double(f.Expected_updated_variables.Z); out.filtZ=double(f.Expected_filtered_variables.Z);
out.updZr=double(f.updated_variables.Z); out.filtZr=double(f.filtered_variables.Z);
out.updXr=double(f.updated_variables.X);
save('../ref/refB2.mat','-struct','out','-v7'); disp('OK');
