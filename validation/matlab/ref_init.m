% RISE filter initialisation options: kf_presample, kf_init_variance, kf_ergodic, kf_user_init
% and diffuse Lyapunov. At commit 1b2edfa: (i) with a unit root and the default initialisation the
% filter returns retcode 23 (any lyapunov_diffuse_factor; lyapunov_algo='schur' is "under revision");
% (ii) kf_user_init with a covariance fails (function issvector does not exist); (iii)
% diffuse_lyapunov_equation only runs with the symmetrize.m patch in this folder.
rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
global NLF_PROBE
out=struct();
cases={'flt_ur','ur';'flt_urms','ur';'flt_ms','ms'};
for c=1:3
  y=readmatrix(['../ref/filter_' cases{c,2} '_data.csv']); db=struct('x',ts('1990Q1',y));
  m=rise(cases{c,1}); m=solve(m); r=struct();
  r.order_var=m.endogenous.name(m.order_var);
  opts=struct('default',{{}},'diffuse10',{{'lyapunov_diffuse_factor',10}},'initvar5',{{'kf_init_variance',5}},...
     'initvar5_pre3',{{'kf_init_variance',5,'kf_presample',3}},'pre3',{{'kf_presample',3}},'nonergodic',{{'kf_ergodic',false}},...
     'user_a0',{{'kf_user_init',{struct('x',0.3,'z',0.2,'w',0.1)}}},'user_cov',{{'kf_user_init',{struct('x',0.3,'z',0.2,'w',0.1),eye(numel(r.order_var))}}});
  if c==3, opts.user_a0={'kf_user_init',{struct('x',0.3,'z',0.2)}}; opts.user_cov={'kf_user_init',{struct('x',0.3,'z',0.2),eye(2)}}; end
  fn=fieldnames(opts);
  for i=1:numel(fn)
    q=struct();
    try
      [~,ll,inc,rc]=filter(m,'data',db,'kf_riccati_tol',0,opts.(fn{i}){:}); q.LogLik=ll; q.Incr=inc; q.retcode=rc;
      NLF_PROBE=[]; filter(m,'data',db,'kf_user_algo',@flt_probe,opts.(fn{i}){:});
      if ~isempty(NLF_PROBE), q.a=NLF_PROBE.syst.a; q.P=NLF_PROBE.syst.P; q.PAI00=NLF_PROBE.syst.PAI00; q.start=NLF_PROBE.syst.start; end
    catch err, q.error=err.message; end
    r.(fn{i})=q;
  end
  out.(cases{c,1})=r;
end
% diffuse Lyapunov (RISE function, with the patch): ordering [x w z] of model flt_ur and other cases
L=struct(); L.T={[0 1 .8;0 1 0;0 0 .8],[0.9 0.5;0 1],[.5 .1;0 .8],[1 0 0;0.3 0.5 0;0.2 0.1 1],[0 1 .5;0 1 0;0 0 .5]};
R=[.3 .1 .1;0 0 .1;.3 0 0]; R2=[.85 .1 .1;0 0 .1;.85 0 0];
L.Q={R*R',eye(2),eye(2),[1 .2 0;.2 1 .1;0 .1 2],R2*R2'}; L.d=[1 10]; L.P=cell(numel(L.T),2); L.rc=zeros(numel(L.T),2);
for i=1:numel(L.T), for j=1:2, [L.P{i,j},L.rc(i,j)]=diffuse_lyapunov_equation(L.T{i},L.Q{i},L.d(j)); end, end
out.diffuse_lyapunov=L;
save('../ref/filter_init.mat','-struct','out','-v7'); disp('OK');
