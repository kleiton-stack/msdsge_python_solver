rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
here=pwd; cd([rise_root '/examples/MarkovSwitching/FoersterRubioRamirezWaggonerZha/QE2016']);
out=struct();
try
  m=rise('rbcs'); [m,rc]=solve(m,'steady_state_unique',true); sol=m.solution;
  out.retcode=rc; out.Tz=cellfun(@full,sol.Tz,'uni',0); out.ss=cellfun(@full,sol.ss,'uni',0); out.Q=full(sol.transition_matrices.Q);
  out.endo={m.endogenous.name}; out.stable=is_stable_system(m);
catch err, out.error=getReport(err); end
try
  m2=rise('rbcs'); [m2,rc2]=solve(m2,'solve_perturbation_type','frwz'); s2=m2.solution;
  out.frwz_retcode=rc2; out.frwz_Tz=cellfun(@full,s2.Tz,'uni',0); out.frwz_ss=cellfun(@full,s2.ss,'uni',0);
catch err, out.frwz_error=getReport(err); end
delete('rbcs*.m'); cd(here);
save('../ref/unique.mat','-struct','out','-v7'); disp('OK');
