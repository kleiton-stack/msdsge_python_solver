% RISE reference for optimal policy: examples from examples/OptimalSimpleRules and
% examples/StochasticReplanning, plus two variants of Canonical (op_canL: regime-switching lead
% coefficient; op_canG: regime-switching commitment probability).
rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
here=pwd; work=tempname; mkdir(work);
ex=[rise_root '/examples/'];
copyfile([ex 'OptimalSimpleRules/Canonical_osr.rs'],work);
copyfile([ex 'StochasticReplanning/Canonical.rs'],work);
copyfile([ex 'StochasticReplanning/usmodel_ms_lc.rs'],work);
copyfile([ex 'StochasticReplanning/Estimation_examples/Canonical_plain.rs'],work);
copyfile([ex 'StochasticReplanning/Estimation_examples/usmodel_sr_switch.rs'],work);
copyfile('op_canL.rs',work); copyfile('op_canG.rs',work);
% Fuhrer-Moore: commitment is a number in the file; generates copies with 1, 0.5 and 0
txt=fileread([ex 'StochasticReplanning/FurherMoore95/fm95_ms_lc.rs']);
vals={'1','0.5','0'}; tags={'fm95_c','fm95_l','fm95_d'};
for k=1:3
  fid=fopen(fullfile(work,[tags{k} '.rs']),'w'); fwrite(fid,strrep(txt,'commitment=1}',['commitment=' vals{k} '}'])); fclose(fid);
end
cd(work);
out=struct();
cases={ % name, file, parameters
 'can_d','Canonical',{'commit_prob',0};
 'can_l','Canonical',{'commit_prob',0.5};
 'can_c','Canonical',{'commit_prob',1};
 'canL_d','op_canL',{'commit_prob',0};
 'canL_l','op_canL',{'commit_prob',0.5};
 'canL_c','op_canL',{'commit_prob',1};
 'canG','op_canG',{};
 'fm95_c','fm95_c',{}; 'fm95_l','fm95_l',{}; 'fm95_d','fm95_d',{};
 'plain','Canonical_plain',{};
 'osr','Canonical_osr',{};
 'sw_d','usmodel_ms_lc',{'ProbabilityOfCommitment',0};
 'sw_l','usmodel_ms_lc',{'ProbabilityOfCommitment',0.5};
 'sw_c','usmodel_ms_lc',{'ProbabilityOfCommitment',1};
 'swsr','usmodel_sr_switch',{}};
% optimal simple rule: estimate maximises welfare over the bounded parameters
try
  me=estimate(rise('Canonical_osr')); pm=me.estimation.posterior_maximization;
  o=struct('mode',pm.mode,'log_lik',pm.log_lik,'log_post',pm.log_post,'log_prior',pm.log_prior,'x0',pm.x0,'f0',pm.f0,'funevals',pm.funevals);
  [w,rcl,V,d]=loss(me); o.loss_welf=full(w); o.loss_d=full(d); o.Tz=full(me.solution.Tz{1});
  out.osr_opt=o;
catch err, out.osr_opt=struct('error',getReport(err,'basic')); end
models=struct();
for k=1:size(cases,1)
  r=struct();
  try
    f=cases{k,2};
    if ~isfield(models,f)
      if strcmp(f,'usmodel_sr_switch'), models.(f)=rise(f,'fix_point_maxiter',5000);
      elseif strncmp(f,'usmodel',7), models.(f)=rise(f,'fix_point_maxiter',5000);
      else, models.(f)=rise(f); end
    end
    m=models.(f); pr=cases{k,3};
    for q=1:2:numel(pr), m=set(m,'parameters',{pr{q},pr{q+1}}); end
    tic; [ms,rc]=solve(m); r.time=toc; r.retcode=rc;
    if rc==0
      sol=ms.solution; r.Tz=cellfun(@full,sol.Tz,'uni',0); r.Q=full(sol.transition_matrices.Q);
      r.ss=cellfun(@full,sol.ss,'uni',0);
      r.endo={ms.endogenous.name}; r.exo={ms.exogenous.name}; r.order_var=ms.order_var;
      r.state_names=ms.endogenous.name(ms.endogenous.is_state);
      r.regimes=ms.markov_chains.regimes;
      r.stable=is_stable_system(ms);
      r.planner_commitment=cellfun(@full,sol.planner.commitment); r.planner_discount=cellfun(@full,sol.planner.discount);
      try
        [w,rcl,V,d]=loss(ms); r.loss_welf=full(w); r.loss_retcode=rcl; r.loss_V=full(V); r.loss_d=full(d);
      catch err, r.loss_error=err.message; end
      if ismember(cases{k,1},{'can_l','fm95_c','osr','sw_l'})
        ii=irf(ms,'irf_periods',20); sh=fieldnames(ii);
        for a=1:numel(sh), vn=fieldnames(ii.(sh{a}));
          for b=1:numel(vn), r.(['irf__' sh{a} '__' vn{b}])=double(ii.(sh{a}).(vn{b})); end, end
      end
    end
  catch err, r.error=getReport(err,'basic'); end
  out.(cases{k,1})=r;
  save(fullfile(here,'..','ref','optimal_policy.mat'),'-struct','out','-v7'); fprintf('feito: %s\n',cases{k,1});
end
cd(here); save('../ref/optimal_policy.mat','-struct','out','-v7'); disp('OK');
