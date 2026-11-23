% RISE references for the SVAR (svar) with and without regime switching.
% Source: examples/VariousModels/TaoZha/Tutorials/SVAR (driver0.m, driver6.m), with feasible beta priors.
rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
raw=load([rise_root '/examples/VariousModels/TaoZha/Tutorials/SVAR/dataraw_allvars.mat']);
vl={'FFR','pi','ygap'}; out=struct(); out.data=100*raw.xdd; out.names=vl;
db=ts('1954Q3',100*raw.xdd,vl); db=pages2struct(db);
var_prior=svar.prior_template(); var_prior.type='sz';
lin0={'a1(1,pi)=0';'a2(1,pi)=0';'a1(1,ygap)=0';'a2(1,ygap)=0';'a2(1,FFR)=0';'a0(2,FFR)=0';'a1(2,FFR)=0';'a2(2,FFR)=0';'a1(2,ygap)=0';'a2(2,ygap)=0';'a1(3,FFR)=0';'a2(3,FFR)=0';'a1(3,pi)=0';'a2(3,pi)=0';'a0(3,pi)+a0(3,FFR)=0'};
nl0={'a0(3,FFR)>=0';'a1(1,FFR)>=0';'a1(1,FFR)<=1'};
% ---- driver0: constant parameters
evalc('sv=estimate(svar(vl,{},2,true),db,{''1960Q1'',''2015Q2''},struct(''var'',var_prior),[lin0;nl0]);');
out.const=dump(sv,vl);
% ---- variances only, 2 states (as in driver6, with feasible beta priors)
mc=struct('name','vol','number_of_states',2,'controlled_parameters',{{'s'}},'endogenous_probabilities',[],'probability_parameters',[]);
sp=struct(); sp.vol_tp_1_2={0.1,0.1,0.05,'beta'}; sp.vol_tp_2_1={0.1,0.2,0.1,'beta'};
lin1={'a0(1,pi)=0';'a0(1,ygap)=0';'a0(2,ygap)=0'};
evalc('sv=estimate(svar(vl,{},2,true,[],mc),db,{''1960Q1'',''2015Q2''},struct(''var'',var_prior,''nonvar'',sp),lin1);');
out.vol=dump(sv,vl);
% ---- driver6: variances in 3 states, Dirichlet prior
mc=struct('name','syncvol','number_of_states',3,'controlled_parameters',{{'s'}},'endogenous_probabilities',[],'probability_parameters',[]);
sp=struct(); sp.dirichlet_1={0.1,'syncvol_tp_1_2',0.2,'syncvol_tp_1_3',0.2}; sp.dirichlet_2={0.1,'syncvol_tp_2_1',0.2,'syncvol_tp_2_3',0.2}; sp.dirichlet_3={0.1,'syncvol_tp_3_1',0.2,'syncvol_tp_3_2',0.2};
evalc('sv=estimate(svar(vl,{},2,true,[],mc),db,{''1960Q1'',''2015Q2''},struct(''var'',var_prior,''nonvar'',sp),[lin0;nl0],''fmincon'');');
out.vol3=dump(sv,vl);
save('../ref/var_ms.mat','-struct','out','-v7'); disp('OK');
function o=dump(sv,vl)
 o=struct(); e=sv.estim_; o.p=e.estim_param; o.pnames=e.links.estimList; o.Y=e.Y; o.X=e.X;
 [ff,~,~,x0]=pull_objective(sv); o.f0=ff(x0);
 [ll,~,~,f]=filter(sv); o.ll=ll;
 g={'filtered','updated','smoothed'};
 for k=1:3, q=f.([g{k} '_regime_probabilities']); fn=fieldnames(q); M=[]; for j=1:numel(fn), M=[M double(q.(fn{j}))]; end; o.([g{k} '_probs'])=M; end
 s=solve(sv); o.A=s.A; o.S0=s.S0; o.B=s.B; o.S=s.S; o.Q=s.Q.Q;
 try, i0=irf(sv,[],12); f=fieldnames(i0); nr=size(double(i0.(f{1}).(vl{1})),2); A=zeros(12,3,3,nr);
   for a=1:3, for b=1:3, A(:,b,a,:)=double(i0.(f{a}).(vl{b})); end, end; o.irf=A; catch me, o.irf_error=me.message; end
end
