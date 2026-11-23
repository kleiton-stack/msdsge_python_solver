rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
y=readmatrix('../ref/filter_ms_data.csv'); db=struct('x',ts('1990Q1',y));
m=rise('flt_ms'); m=solve(m);
[f,LogLik]=filter(m,'data',db,'kf_riccati_tol',0);
out=struct('LogLik',LogLik);
out.smo_z=double(f.Expected_smoothed_variables.z); out.smo_x=double(f.Expected_smoothed_variables.x);
out.smo_e=double(f.Expected_smoothed_shocks.e); out.smo_u=double(f.Expected_smoothed_shocks.u);
out.upd_z=double(f.Expected_updated_variables.z); out.filt_z=double(f.Expected_filtered_variables.z);
out.smo_z_r=double(f.smoothed_variables.z); out.smo_e_r=double(f.smoothed_shocks.e);
out.smo1=double(f.smoothed_regime_probabilities.regime_1);
save('../ref/smooth_ms.mat','-struct','out','-v7'); disp('OK');
