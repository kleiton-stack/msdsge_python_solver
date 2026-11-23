rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
y=readmatrix('../ref/filter_ms_data.csv'); db=struct('x',ts('1990Q1',y));
out=struct();
try
  m=rise('est_ms'); m=set(m,'data',db,'kf_riccati_tol',0);
  [m0,ff]=estimate(m,'optimset',struct('MaxIter',0,'MaxFunEvals',1,'Display','off'));
catch err, out.est_error=getReport(err); m0=[]; end
try
  pts=[0.05 0.10 0.1 0.9 0.1 0.2 1.5; 0.08 0.15 0.12 0.8 0.0 0.3 1.2; 0.03 0.25 0.2 0.93 -0.2 0.15 2.0];
  if isempty(m0), m0=m; end
  out.names={m0.estimation.priors.name};
  try, out.prior_dist={m0.estimation.priors.prior_distrib}; out.prior_mean=[m0.estimation.priors.prior_mean]; out.prior_stdev=[m0.estimation.priors.prior_stdev]; catch e1, out.pe=e1.message; end
  try, out.lb=[m0.estimation.priors.lower_bound]; out.ub=[m0.estimation.priors.upper_bound]; catch, end
  res=zeros(size(pts,1),3);
  for k=1:size(pts,1)
    [lp,ll,lpr,~,rc]=log_posterior_kernel(m0,pts(k,:)'); res(k,:)=[lp ll sum(lpr(:))]; out.rc(k)=rc; out.lpr{k}=lpr;
  end
  out.pts=pts; out.res=res;
catch err, out.error=getReport(err); end
save('../ref/post.mat','-struct','out','-v7'); disp('OK');
