% Hessian at the mode: the one stored by RISE's estimate against finite differences of RISE's own objective.
rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
y=readmatrix('../ref/filter_ms_data.csv'); db=struct('x',ts('1990Q1',y));
here=pwd; wd=tempname; mkdir(wd); copyfile('est_ms2.rs',wd); cd(wd);
out=struct();
try
  rng(1);
  m=rise('est_ms2'); m=set(m,'data',db,'kf_riccati_tol',0);
  m=estimate(m);
  [objective,lb,ub,x0,SIG]=pull_objective(m);
  out.x0=x0; out.f0=objective(x0); out.hessian_estimate=m.estimation.posterior_maximization.hessian;
  d=numel(x0); h=1e-3*max(abs(x0),1e-2); out.h=h;
  H=zeros(d);
  for i=1:d
    ei=zeros(d,1); ei(i)=h(i);
    H(i,i)=(objective(x0+ei)-2*out.f0+objective(x0-ei))/h(i)^2;
    for j=1:i-1
      ej=zeros(d,1); ej(j)=h(j);
      H(i,j)=(objective(x0+ei+ej)-objective(x0+ei-ej)-objective(x0-ei+ej)+objective(x0-ei-ej))/(4*h(i)*h(j)); H(j,i)=H(i,j);
    end
  end
  out.hessian_central=H;
  try, out.hessian_fd=utils.hessian.numerical(objective,x0,'fd'); catch e1, out.fd_err=e1.message; end
  try, out.hessian_opg=utils.hessian.numerical(objective,x0,'opg'); catch e1, out.opg_err=e1.message; end
catch err, out.error=getReport(err,'extended','hyperlinks','off'); disp(out.error); end
cd(here); save('../ref/est_ms2_hess.mat','-struct','out','-v7'); disp('OK');
