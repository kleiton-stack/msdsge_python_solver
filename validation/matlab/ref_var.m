% RISE references for the reduced-form VAR (rfvar): estimation, identification and analyses.
% Source examples: examples/RFVAR_identification/howto.m and VariousModels/HildeBjornland/tut*.m
rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
out=struct();
[num,txt]=xlsread([rise_root '/examples/RFVAR_identification/Smets_Wouters_data.xlsx']);
vnames=strrep(txt,' ',''); num=num(:,2:end);
out.sw_data=num; out.sw_names=vnames;
data=ts('1947q3',num,vnames); data=pages2struct(data);
endog={'robs','dy','labobs','pinfobs','dw','dc','dinve'};
% ---- 1. 7-variable VAR(3) from howto.m
rv1=estimate(rfvar(endog,{},3,true),data);
p=rv1.estim_.estim_param; s=solve(rv1);
o=struct(); o.endog=endog; o.p=p; o.pnames=rv1.parameters; o.B=s.B; o.S=s.S; o.ll=filter(rv1); o.Y=rv1.estim_.Y; o.X=rv1.estim_.X;
i0=irf(rv1,[],12); f=fieldnames(i0); A=zeros(12,7,7); for a=1:7, for b=1:7, A(:,b,a)=double(i0.(f{a}).(endog{b})); end, end; o.irf=A;
vd=variance_decomposition(rv1); for b=1:7, x=double(vd.conditional.(endog{b})); o.vd(:,:,b)=x(1:40,:); o.vd_inf(b,:)=double(vd.infinity.(endog{b})); end
hd=historical_decomposition(rv1); for b=1:7, o.hd(:,:,b)=double(hd.(endog{b})); end; o.hd_names=hd.robs.varnames;
ss=structural_shocks(rv1); for b=1:7, o.shocks(:,b)=double(ss.(f{b})); end
[C,R]=autocov(rv1); o.autocov=C; o.autocorr=R;
fk=forecast(rv1,data,[],[],8,false); for b=1:7, o.fk(:,b)=double(fk.(endog{b})); end
fk=forecast(rv1,data,'2000Q1',[],8,false); for b=1:7, o.fk_2000(:,b)=double(fk.(endog{b})); end
cond=struct(); cond.robs={'2000Q1','2000Q4'}; cond.dy={'2000Q1','2000Q2'};
fk=forecast(rv1,data,'2000Q1',[],8,false,[],cond); for b=1:7, o.fk_cond(:,b)=double(fk.(endog{b})); end
% sign restrictions from howto.m (random draws: only properties are compared)
sn={'mp','ad','as','ls'};
restr={'robs{0}@mp','+';'dy@mp','-';'pinfobs@mp','-';'robs{0}@ad','+';'pinfobs@ad','+';'dy@ad','+';'dy@as','+';'pinfobs@as','-';'robs@ls','-';'dy@ls','+';'labobs@ls','+';'dw@ls','-'};
rng(1); [Rf,id]=identification(rv1,restr,sn,true,6000); o.sign_R=Rf(p); o.sign_ident=[id.isexact id.isover id.isunder];
out.sw7=o;
% ---- 2. 3-variable VAR(2): exact identification (short run, long run, mixed)
e3={'robs','dy','pinfobs'}; rv=estimate(rfvar(e3,{},2,true),data); p=rv.estim_.estim_param; s=solve(rv);
o=struct(); o.endog=e3; o.B=s.B; o.S=s.S; o.ll=filter(rv);
sn={'s1','s2','s3'}; nrm={'robs{0}@s1','+';'dy{0}@s2','+';'pinfobs{0}@s3','+'};
cases={ [nrm;{'robs{0}@s2',0;'robs{0}@s3',0;'dy{0}@s3',0}], ...
        [nrm;{'robs{0}@s2',0;'robs{0}@s3',0;'dy{inf}@s3',0}], ...
        [nrm;{'robs{inf}@s2',0;'robs{inf}@s3',0;'dy{inf}@s3',0}], ...
        [nrm;{'robs{inf}@s2',0;'robs{inf}@s3',0;'dy{0}@s3',0}], ...
        {'robs{0}@s1','-';'dy{0}@s2','-';'pinfobs{0}@s3','-';'robs{0}@s2',0;'robs{0}@s3',0;'dy{1}@s3',0}, ...
        [nrm;{'dy{0}@s1',0;'pinfobs{0}@s1',0;'dy{inf}@s2',0}]};
for k=1:numel(cases)
  [Rf,id]=identification(rv,cases{k},sn,true,6000); o.(sprintf('R%d',k))=Rf(p); o.(sprintf('ident%d',k))=[id.isexact id.isover id.isunder];
end
[Rf,id]=identification(rv,cases{4},sn,true,6000);
i0=irf(rv,sn,12,[],Rf); A=zeros(12,3,3); for a=1:3, for b=1:3, A(:,b,a)=double(i0.(sn{a}).(e3{b})); end, end; o.irf4=A;
vd=variance_decomposition(rv,[],Rf,sn); for b=1:3, x=double(vd.conditional.(e3{b})); o.vd4(:,:,b)=x(1:40,:); o.vd4_inf(b,:)=double(vd.infinity.(e3{b})); end
hd=historical_decomposition(rv,[],Rf,sn); for b=1:3, o.hd4(:,:,b)=double(hd.(e3{b})); end; o.hd4_names=hd.robs.varnames;
ss=structural_shocks(rv,[],Rf,sn); for b=1:3, o.shocks4(:,b)=double(ss.(sn{b})); end
[Rf,id]=identification(rv,'choleski',{'pinfobs','robs','dy'}); o.R_chol_order=Rf(p);
out.sw3=o;
% ---- 3. exogenous variable, no constant, linear restrictions (sample 1960Q1-1999Q4)
data.tt=ts('1947q3',(1:230)'/100); o=struct(); rg={'1960Q1','1999Q4'};
rv=estimate(rfvar(e3,{'tt'},2,true),data,rg); s=solve(rv); o.B_exo=s.B; o.S_exo=s.S; o.ll_exo=filter(rv); o.names_exo=rv.parameters;
rv=estimate(rfvar(e3,{},2,false),data,rg); s=solve(rv); o.B_noconst=s.B; o.S_noconst=s.S; o.ll_noconst=filter(rv);
lr={'b1(1,dy)=0';'b2(1,dy)=0';'b1(2,robs)+b1(2,pinfobs)=0'};
rv=estimate(rfvar(e3,{},2,true),data,rg,[],lr); s=solve(rv); o.B_linres=s.B; o.S_linres=s.S; o.ll_linres=filter(rv);
% ---- 4. BVAR with the sz prior (the only BVAR prior that runs in this RISE version)
pr=rfvar.prior_template(); pr.type='sz';
rv=estimate(rfvar(e3,{},2,true),data,rg,struct('var',pr)); s=solve(rv); o.B_sz=s.B; o.S_sz=s.S; o.ll_sz=filter(rv);
rng(2); dr=rv.estim_.sampler(4000); o.sz_draw_mean=mean(dr,2); o.sz_draw_std=std(dr,0,2);
pr.L1=0.2; pr.L3=2; pr.L5=1; pr.L6=1; pr.L7=0.5;
rv=estimate(rfvar(e3,{},2,true),data,rg,struct('var',pr)); s=solve(rv); o.B_sz2=s.B; o.S_sz2=s.S; o.ll_sz2=filter(rv);
pr=rfvar.prior_template(); pr.type='sz';
try, rv=estimate(rfvar(e3,{},2,true),data,rg,struct('var',pr),lr); s=solve(rv); o.B_sz_linres=s.B; o.S_sz_linres=s.S; catch me, o.sz_linres_error=me.message; end
try, rv=estimate(rfvar(e3,{'tt'},2,true),data,rg,struct('var',pr)); s=solve(rv); o.B_sz_exo=s.B; o.S_sz_exo=s.S; catch me, o.sz_exo_error=me.message; end
types={'minnesota','jeffrey','nw','normal-wishart','inw'}; o.broken={};
for k=1:numel(types)
  pr=rfvar.prior_template(); pr.type=types{k};
  try, evalc('rv=estimate(rfvar(e3,{},2,true),data,rg,struct(''var'',pr));'); catch me, o.broken=[o.broken;{types{k},me.message}]; end
end
out.misc=o;
save('../ref/var_rf.mat','-struct','out','-v7'); disp('OK');
