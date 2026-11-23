% RISE references for msdsge.occbin (occasionally binding constraints).
%  one: examples/occbin/oneConstraint (ZLB), simulation and IRFs with simul_honor_constraints
%  two: examples/occbin/multipleConstraints (collateral and ZLB)
%  sw : globdynzlb.rs with the line "? R>=1" and simul_honor_constraints_through_switch
rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
here=pwd; ex=[rise_root '/examples']; out=struct();

%% one constraint
w=tempname; mkdir(w); copyfile(fullfile(ex,'occbin','oneConstraint','*'),w); cd(w);
m=rise('occbin');
p=baselineParameters(); p.r_zlb=1; p.bind_ocb_1=0; p.bind_ocb_2=1; p.ocb_tp_1_2=0; p.ocb_tp_2_1=0;
m=set(m,'parameters',p,'steady_state_file','ssfile_full');
m=set(m,'solve_occbin',1,'steady_state_imposed',true,'steady_state_unique',true);
[m,rc]=solve(m);
out.one=collect(m,rc,1,300,[5 30 60]);
cd(here);

%% two constraints
w=tempname; mkdir(w); copyfile(fullfile(ex,'occbin','multipleConstraints','*'),w); cd(w);
m=rise('ig2017');
m=set(m,'parameters',paramfile(),'steady_state_file','ssfile');
restr_map=struct('debt',1,'zlb',2);
[m,rc]=solve(m,'solve_occbin',{1,restr_map,m.markov_chains.regimes},'steady_state_unique',true,'steady_state_imposed',true);
out.two=collect(m,rc,1900,300,[5 20]);
out.two.regimes=m.markov_chains.regimes;
cd(here);

%% constraint honoured through regime switching
w=tempname; mkdir(w); cd(w);
txt=fileread(fullfile(ex,'MarkovSwitching','GavinKeenRichterThrockmorton','globdynzlb.rs'));
txt=strrep(txt,'	W = chi*N^eta*C^sigma;',sprintf('	? R>=1;\n	W = chi*N^eta*C^sigma;'));
txt=strrep(txt,'sigmabeta,0.0025;','sigmabeta,0.01;');   % larger shock, so that the constraint binds often
fid=fopen('gk.rs','w'); fwrite(fid,txt); fclose(fid);
m=rise('gk','rise_flags',struct('exogenous_switching',true));
[m,rc]=solve(m,'steady_state_imposed',true);
r=struct(); r.retcode=rc; sol=m.solution;
r.Tz=cellfun(@full,sol.Tz,'uni',0); r.ss=cellfun(@full,sol.ss,'uni',0); r.Q=full(sol.transition_matrices.Q);
r.endo=m.endogenous.name; r.exo=m.exogenous.name; r.order_var=m.order_var;
rng(7);
s=simulate(m,'simul_periods',2000,'simul_burn',0,'simul_honor_constraints',true,'simul_honor_constraints_through_switch',true);
r.sim=tomat(s,r.endo); r.shocks=tomat(s,r.exo); r.shocks=r.shocks(2:end,:); r.regime=double(s.regime); r.regime=r.regime(2:end);
out.sw=r;
cd(here);

save('../ref/occbin.mat','-struct','out','-v7'); disp('OK');

function r=collect(m,rc,seed,T,sizes)
r=struct(); r.retcode=rc; sol=m.solution;
r.Tz=full(sol.Tz{1}); r.ss=full(sol.ss{1});
r.endo=m.endogenous.name; r.exo=m.exogenous.name; r.order_var=m.order_var;
rng(seed);
s=simulate(m,'simul_periods',T,'simul_burn',0,'simul_honor_constraints',true);
r.sim=tomat(s,r.endo);                 % row 1 = initial condition (steady state)
r.shocks=tomat(s,r.exo); r.shocks=r.shocks(2:end,:);
r.regime=double(s.regime); r.regime=r.regime(2:end);
r.irf_sizes=sizes;
for k=1:numel(sizes)
  M=[set(m,'irf_shock_sign',-sizes(k)),set(m,'irf_shock_sign',sizes(k))];
  ir=irf(M,'simul_honor_constraints',true,'irf_periods',40);
  A=zeros(41,2,numel(r.endo),numel(r.exo));  % period x sign(-,+) x variable x shock
  for a=1:numel(r.exo), for b=1:numel(r.endo), A(:,:,b,a)=double(ir.(r.exo{a}).(r.endo{b})); end, end
  r.(sprintf('irf_%d',sizes(k)))=A;
end
end

function X=tomat(s,names)
X=[]; for i=1:numel(names), X=[X,double(s.(names{i}))]; end %#ok<AGROW>
end
