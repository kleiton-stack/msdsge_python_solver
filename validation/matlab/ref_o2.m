rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
out=struct(); names={'o2c','o2'};
for q=1:2
  r=struct();
  try
    m=rise(names{q},'max_deriv_order',2); [m,rc]=solve(m,'solve_order',2); sol=m.solution;
    r.retcode=rc; r.fields=fieldnames(sol); r.Tz=cellfun(@full,sol.Tz,'uni',0); r.Tzz=cellfun(@full,sol.Tzz,'uni',0);
    r.ss=cellfun(@full,sol.ss,'uni',0); r.endo={m.endogenous.name}; r.states=m.endogenous.name(m.endogenous.is_state);
    r.order_var=m.order_var;
  catch err, r.error=getReport(err); end
  out.(names{q})=r;
end
save('../ref/o2.mat','-struct','out','-v7'); disp('OK');
