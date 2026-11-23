% Endogenous probabilities (point at which RISE evaluates Q) and theoretical moments by order
rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
here=pwd; work=tempname; mkdir(work); copyfile('o2x.rs',work); copyfile('o2y.rs',work); copyfile('o2w.rs',work); copyfile('o2c.rs',work); copyfile('o2.rs',work); cd(work);
out=struct(); fl={'Tz','Tzz','Tzzz'};
try
  m0=rise('o2x','max_deriv_order',3); [m,rc]=solve(m0,'solve_order',3); sol=m.solution; r=struct(); r.retcode=rc;
  for k=1:3, r.(fl{k})=cellfun(@full,sol.(fl{k}),'uni',0); end
  r.ss=cellfun(@full,sol.ss,'uni',0); r.endo=m.endogenous.name; r.state_list=create_state_list(m,1); r.Q=full(sol.transition_matrices.Q);
  r.solfields=fieldnames(sol); r.tmfields=fieldnames(sol.transition_matrices);
  out.o2x=r;
catch err, out.o2x_error=getReport(err); disp(out.o2x_error); end
% o2y and o2w: same equations as o2x, probabilities in (C,K) and in (A,C); 1st order only
names={'o2y','o2w'};
for q=1:2
  try
    m=rise(names{q}); [m,rc]=solve(m); sol=m.solution; r=struct(); r.retcode=rc;
    r.Tz=cellfun(@full,sol.Tz,'uni',0); r.ss=cellfun(@full,sol.ss,'uni',0); r.Q=full(sol.transition_matrices.Q);
    r.endo=m.endogenous.name; r.state_list=create_state_list(m,1); r.order_var=m.order_var; out.(names{q})=r;
  catch err, out.([names{q} '_error'])=getReport(err); disp(getReport(err)); end
end
names={'o2c','o2'};
for q=1:2
  try
    m0=rise(names{q},'max_deriv_order',3); r=struct();
    for k=1:3
      m=solve(m0,'solve_order',k); [ac,rc]=theoretical_autocovariances(m,'autocov_ar',2); r.(sprintf('ac%d',k))=ac; r.(sprintf('rc%d',k))=rc;
    end
    out.(['mom_' names{q}])=r;
  catch err, out.(['mom_' names{q} '_error'])=getReport(err); disp(getReport(err)); end
end
cd(here); save('../ref/pert2.mat','-struct','out','-v7'); disp('OK');
