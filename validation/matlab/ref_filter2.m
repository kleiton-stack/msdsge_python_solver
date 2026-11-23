rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
y=readmatrix('../ref/filter_ms_data.csv'); db=struct('x',ts('1990Q1',y));
m=rise('flt_ms'); m=solve(m);
d=filter(rise.empty(0)); disp(d);
out=struct();
try
 [~,LogLik,Incr]=filter(m,'data',db,'kf_riccati_tol',0); out.tol0=struct('LogLik',LogLik,'Incr',Incr);
catch e, disp(e.message); end
save('../ref/filter_ms_tol0.mat','-struct','out','-v7'); disp('OK');
