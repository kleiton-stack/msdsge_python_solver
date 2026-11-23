% RISE references for sensitivity.py: quasi_monte_carlo, hdmr (Ishigami, Sobol g) and mcf (trinity.rs)
rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
ex=[rise_root '/examples/'];
addpath([ex 'HighDimensionalModelRepresentation']);
here=pwd; work=tempname; mkdir(work); copyfile([ex 'MonteCarloFiltering/*'],work); cd(work);
out=struct();
funs={'ishigami','satelli_sobol95'};
for k=1:2
  r=struct();
  try
    [objective,bounds]=feval(funs{k}); lb=bounds(:,1); ub=bounds(:,2); npar=numel(lb);
    param_names=strcat('pp_',num2str((1:npar)'));
    N=2^10; rng(1);
    theta=quasi_monte_carlo.sobol(lb,ub,N,false); r.theta=theta; r.theta_size=size(theta);
    f=objective(theta); r.f=f;
    for po=[4 8]
      for opt=[0 1]
        tag=sprintf('p%d_o%d',po,opt);
        try
          obj=hdmr({f,theta},param_names,bounds,2,po,logical(opt));
          obj=estimate(obj,false);
          s=struct(); w=warning('off','all'); pr=properties(obj);
          for q=1:numel(pr)
            v=obj.(pr{q});
            if isnumeric(v)||islogical(v)||ischar(v), s.(pr{q})=v;
            elseif iscell(v), try, s.(pr{q})=v; catch, end
            elseif isstruct(v), s.(pr{q})=v; end
          end
          warning(w); s.props=pr;
          try, s.meta=metamodel(obj,theta(:,1:20)); catch e2, s.meta_error=getReport(e2); end
          r.(tag)=s;
        catch err, r.([tag '_error'])=getReport(err); disp(getReport(err)); end
      end
    end
  catch err, r.error=getReport(err); disp(getReport(err)); end
  out.(funs{k})=r;
end
try
  lbq=[0;-1;2]; ubq=[1;1;5];
  out.qmc_sobol=quasi_monte_carlo.sobol(lbq,ubq,16,false);
  out.qmc_halton=quasi_monte_carlo.halton(lbq,ubq,16);
  rng(2); out.qmc_lhs=quasi_monte_carlo.latin_hypercube(lbq,ubq,16);
catch err, out.qmc_error=getReport(err); disp(getReport(err)); end
try
  m=rise('trinity');
  parameter_names={m.estimation.priors.name}; lb=vertcat(m.estimation.priors.lower_bound); ub=vertcat(m.estimation.priors.upper_bound);
  out.mcf_names=parameter_names; out.mcf_lb=lb; out.mcf_ub=ub;
  rng(3);
  obj=mcf(@(x)is_has_solution(m,parameter_names,x),500,lb,ub,parameter_names,'latin_hypercube');
  out.mcf_samples=obj.samples; out.mcf_is_behaved=obj.is_behaved; out.mcf_props=properties(obj);
  rng(4);
  obj2=mcf(@(x)price_puzzle(m,parameter_names,x),500,lb,ub,parameter_names,'latin_hypercube');
  out.mcf2_samples=obj2.samples; out.mcf2_is_behaved=obj2.is_behaved;
  out.mcf_methods=methods(obj);
  % stability on the same samples
  st=false(1,size(obj.samples,2)); rc=zeros(1,size(obj.samples,2));
  for q=1:size(obj.samples,2)
    p=struct(); for w=1:numel(parameter_names), p.(parameter_names{w})=obj.samples(w,q); end
    [mm,rc(q)]=solve(m,'parameters',p); if rc(q)==0, st(q)=is_stable_system(mm); end
  end
  out.mcf_retcode=rc; out.mcf_stable=st;
catch err, out.mcf_error=getReport(err); disp(getReport(err)); end
cd(here); save('../ref/sens.mat','-struct','out','-v7'); disp('OK');
