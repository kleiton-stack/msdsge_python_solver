% Generates RISE reference solutions for the FWZ example (3 parameterisations, 4 solvers)
rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
here = pwd; work = tempname; mkdir(work);
copyfile([rise_root '/examples/MarkovSwitching/FarmerWaggonerZha2010/fwz10_nasty.rs'], work);
cd(work);
m0 = rise('fwz10_nasty');
pp=struct(); pp.delta_a_1=0; pp.delta_a_2=0; pp.betta_a_1=1; pp.betta_a_2=1;
pp.rho_a_1=.9; pp.rho_a_2=.9; pp.phi_a_1=.5; pp.phi_a_2=.8; pp.a_tp_1_2=1-.8; pp.a_tp_2_1=1-.9;
pp(2)=pp; pp(2).delta_a_1=-.7; pp(2).delta_a_2=.4; pp(2).rho_a_1=0; pp(2).rho_a_2=0;
pp(2).a_tp_1_2=1-1; pp(2).a_tp_2_1=1-.64;
pp(3)=pp(2); pp(3).delta_a_2=-.2; pp(3).phi_a_1=.2; pp(3).phi_a_2=.4; pp(3).a_tp_1_2=1-.9; pp(3).a_tp_2_1=1-.8;
solvers={'mnk','mn','mfi','fwz'};
out=struct();
for ii=1:3
  for s=1:numel(solvers)
    key=sprintf('p%d_%s',ii,solvers{s});
    try
      [m,retcode]=solve(m0,'parameters',pp(ii),'solver',solvers{s});
      r=struct('retcode',retcode);
      if retcode==0
        sol=m.solution;
        r.fields=fieldnames(sol);
        r.Tz=sol.Tz; r.ss=sol.ss; r.Q=full(sol.transition_matrices.Q);
        r.order_var=m.order_var; r.endo={m.endogenous.name}; r.exo={m.exogenous.name};
        r.state_cols=m.locations.after_solve.z;
        r.stable=is_stable_system(m);
      end
      out.(key)=r;
    catch err
      out.(key)=struct('error',err.message);
    end
  end
end
cd(here);
save('../ref/fwz.mat','-struct','out','-v7');
disp('OK');
