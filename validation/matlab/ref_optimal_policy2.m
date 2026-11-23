% Complement to ref_optimal_policy.m: fm95_ms_lc.rs and Canonical_plain.rs use "model(linear)",
% which the local RISE (commit 1b2edfa) does not accept (error in parser.capture_equations). Here the
% files are copied with "model(linear)" replaced by "model" (the models are already linear).
rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
here=pwd; work=tempname; mkdir(work);
ex=[rise_root '/examples/StochasticReplanning/'];
txt=strrep(fileread([ex 'FurherMoore95/fm95_ms_lc.rs']),'model(linear)','model');
vals={'1','0.5','0'}; tags={'fm95_c','fm95_l','fm95_d'};
for k=1:3
  fid=fopen(fullfile(work,[tags{k} '.rs']),'w'); fwrite(fid,strrep(txt,'commitment=1}',['commitment=' vals{k} '}'])); fclose(fid);
end
txt=strrep(fileread([ex 'Estimation_examples/Canonical_plain.rs']),'model(linear)','model');
fid=fopen(fullfile(work,'plain.rs'),'w'); fwrite(fid,txt); fclose(fid);
cd(work); out=struct(); names=[tags {'plain'}];
for k=1:numel(names)
  r=struct();
  try
    m=rise(names{k}); [ms,rc]=solve(m); r.retcode=rc;
    if rc==0
      sol=ms.solution; r.Tz=cellfun(@full,sol.Tz,'uni',0); r.Q=full(sol.transition_matrices.Q);
      r.endo={ms.endogenous.name}; r.exo={ms.exogenous.name}; r.order_var=ms.order_var;
      r.state_names=ms.endogenous.name(ms.endogenous.is_state); r.stable=is_stable_system(ms);
      ii=irf(ms,'irf_periods',20); sh=fieldnames(ii);
      for a=1:numel(sh), vn=fieldnames(ii.(sh{a}));
        for b=1:numel(vn), r.(['irf__' sh{a} '__' vn{b}])=double(ii.(sh{a}).(vn{b})); end, end
    end
  catch err, r.error=getReport(err,'basic'); end
  out.(names{k})=r;
end
cd(here); save('../ref/optimal_policy2.mat','-struct','out','-v7'); disp('OK');
