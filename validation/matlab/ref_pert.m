% Higher-order perturbation references (solution, simulation, IRFs) for tests/test_perturbation.py
rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
here=pwd; work=tempname; mkdir(work);
ex=[rise_root '/examples/'];
copyfile('o2.rs',work); copyfile('o2c.rs',work); copyfile('o2e.rs',work); copyfile('o2r.rs',work);
copyfile([ex 'Nonlinear3rdOrder/Binning/small_nk.rs'],work);
copyfile([ex 'MarkovSwitching/FoersterRubioRamirezWaggonerZha/frwz_nk.rs'],work);
cd(work);
out=struct();
cases={'o2c',5,{}; 'o2',5,{}; 'o2e',3,{}; 'o2r',3,{}; 'small_nk',5,{}; 'frwz_nk',3,{}; 'frwz_nk',3,{'steady_state_unique',true}};
tags={'o2c','o2','o2e','o2r','small_nk','frwz_nk','frwz_nk_unique'};
fl={'Tz','Tzz','Tzzz','Tzzzz','Tzzzzz'};
for q=1:size(cases,1)
  r=struct(); name=cases{q,1}; kmax=cases{q,2};
  try
    m0=rise(name,'max_deriv_order',kmax);
    r.time=nan(1,kmax); r.retcode=nan(1,kmax);
    for k=1:kmax
      tic; [m,rc]=solve(m0,'solve_order',k,cases{q,3}{:}); r.time(k)=toc; r.retcode(k)=rc;
      tic; [m,rc]=solve(m0,'solve_order',k,cases{q,3}{:}); r.time(k)=min(r.time(k),toc);
    end
    sol=m.solution;
    for k=1:kmax, r.(fl{k})=cellfun(@full,sol.(fl{k}),'uni',0); end
    r.ss=cellfun(@full,sol.ss,'uni',0); r.endo=m.endogenous.name; r.order_var=m.order_var;
    r.state_list=create_state_list(m,1); r.Q=full(sol.transition_matrices.Q);
    out.(tags{q})=r; models.(tags{q})=m;
  catch err, r.error=getReport(err); out.(tags{q})=r; disp(r.error); end
end
% simulations: the shocks drawn by RISE are returned in the database and reapplied in Python
sims={'o2c','o2','o2e','small_nk'};
for q=1:numel(sims)
  if ~isfield(models,sims{q}), continue, end
  m=models.(sims{q}); names=[m.endogenous.name, m.exogenous.name]; S=struct();
  for ord=1:3
    for pr=0:1
      for fixreg=0:1
        if m.markov_chains.regimes_number==1 && fixreg==0, continue, end
        tag=sprintf('o%d_p%d_r%d',ord,pr,fixreg); x=struct();
        try
          rng(7); args={'simul_periods',80,'simul_burn',0,'simul_order',ord,'simul_pruned',logical(pr)};
          if fixreg, args=[args,{'simul_regime',1}]; end
          [s,states,rc]=simulate(m,args{:});
          x.retcode=rc; x.names=names; x.data=nan(numel(double(s.(names{1}))),numel(names));
          for i=1:numel(names), x.data(:,i)=double(s.(names{i})); end
          x.regime=double(s.regime); x.states=states;
        catch err, x.error=getReport(err); disp(x.error); end
        S.(tag)=x;
      end
    end
  end
  out.(['sim_' sims{q}])=S;
end
% regime-specific and generalised IRFs at order 3
irfs={'o2c','o2','small_nk'};
for q=1:numel(irfs)
  if ~isfield(models,irfs{q}), continue, end
  m=models.(irfs{q}); x=struct();
  try
    rng(3); my=irf(m,'irf_periods',12,'simul_order',3); x.shocks=fieldnames(my); sh=x.shocks{1}; v=m.endogenous.name;
    x.fields=fieldnames(my.(sh));
    for i=1:numel(v), x.(['irf_' v{i}])=double(my.(sh).(v{i})); end
    rng(3); my=irf(m,'irf_periods',12,'simul_order',3,'simul_pruned',true);
    for i=1:numel(v), x.(['irfp_' v{i}])=double(my.(sh).(v{i})); end
    rng(3); my=irf(m,'irf_periods',12,'simul_order',3,'irf_type','girf','irf_draws',200,'irf_regime_specific',false);
    for i=1:numel(v), x.(['girf_' v{i}])=double(my.(sh).(v{i})); end
    rng(3); my=irf(m,'irf_periods',12,'simul_order',1);
    for i=1:numel(v), x.(['irf1_' v{i}])=double(my.(sh).(v{i})); end
  catch err, x.error=getReport(err); disp(x.error); end
  out.(['irf_' irfs{q}])=x;
end
cd(here); save('../ref/pert.mat','-struct','out','-v7'); disp('OK');
