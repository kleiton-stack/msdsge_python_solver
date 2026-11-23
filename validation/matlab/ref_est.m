% Estimation reference: mode, Laplace, mh_sampler draws and MDDs from RISE (model est_ms2).
rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
y=readmatrix('../ref/filter_ms_data.csv'); db=struct('x',ts('1990Q1',y));
here=pwd; wd=tempname; mkdir(wd); copyfile('est_ms2.rs',wd); cd(wd);
out=struct();
try
  rng(1);
  m=rise('est_ms2'); m=set(m,'data',db,'kf_riccati_tol',0);
  tic; m=estimate(m); out.time_estimate=toc;
  pm=m.estimation.posterior_maximization;
  fn=fieldnames(pm);
  for k=1:numel(fn)
    v=pm.(fn{k}); if isnumeric(v)||islogical(v), out.(['pm_' fn{k}])=double(v); end
  end
  out.names={m.estimation.priors.name};
  [objective,lb,ub,x0,SIG]=pull_objective(m);
  out.lb=lb; out.ub=ub; out.x0=x0; out.SIG=SIG; out.f_x0=objective(x0);
  [lp,ll,lpr]=log_posterior_kernel(m,x0); out.kernel_x0=[lp ll sum(lpr(:))];
catch err, out.est_error=getReport(err,'extended','hyperlinks','off'); disp(out.est_error); end
cd(here); save('../ref/est_ms2.mat','-struct','out','-v7');
try
  cd(wd); rng(2);
  SIG=utils.cov.nearest(SIG);
  opts=struct('burnin',2000,'N',10000,'thin',1,'nchain',2,'verbose',5000);
  tic; R=mh_sampler(objective,lb,ub,opts,x0,SIG); out.time_mh=toc;
  nc=size(R.pop,1); N=size(R.pop,2); d=numel(x0);
  X=zeros(nc,N,d); F=zeros(nc,N);
  for c=1:nc, for i=1:N, X(c,i,:)=R.pop(c,i).x; F(c,i)=R.pop(c,i).f; end, end
  out.mh_x=X; out.mh_f=F; out.mh_bestf=R.bestf; out.mh_bestx=R.bestx;
  try, out.mh_stats=R.stats; catch, end
  cd(here); save('../ref/est_ms2.mat','-struct','out','-v7'); cd(wd);
  algos={'mhm','swz','mueller','bridge','is','ris','cj','laplace','laplace_mcmc'};
  out.mdd_algos=algos; out.mdd=nan(1,numel(algos)); out.mdd_time=nan(1,numel(algos)); out.mdd_err=cell(1,numel(algos));
  for k=1:numel(algos)
    try
      rng(10+k); o=struct('log_post_kern',objective,'L',5000,'algorithm',algos{k});
      tic; out.mdd(k)=mcmc_mdd(R.pop,lb,ub,o); out.mdd_time(k)=toc;
    catch e2, out.mdd_err{k}=e2.message; disp(e2.message); end
  end
catch err, out.mh_error=getReport(err,'extended','hyperlinks','off'); disp(out.mh_error); end
cd(here); save('../ref/est_ms2.mat','-struct','out','-v7'); disp('OK');
