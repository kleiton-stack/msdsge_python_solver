% Reference for the RISE sigma-point filters on 2nd-order models (one observable, C).
% The default filter for solve_order>1 is switching_divided_difference_filter, which does not run
% on R2024a (syntax error in the p-code); UKF and CKF are selected with 'kf_user_algo'.
% nlf_probe captures the structure that dsge.filter hands to the filter; nlf_logff logs the
% calls to the law of motion (sigma points) in the first 4 periods.
rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
global NLF_PROBE NLF_LOG
out=struct();
for nm={'o2c','o2','o2m'}
  y=readmatrix(['../ref/nlf_' nm{1} '_data.csv']); db=struct('C',ts('1990Q1',y));
  m=rise([nm{1} 'f'],'max_deriv_order',2); m=solve(m,'solve_order',2);
  r=struct();
  try, [~,ll]=filter(m,'data',db); r.default_ll=ll; catch err, r.default_error=err.message; end
  [f,LogLik]=filter(m,'data',db,'kf_user_algo',@nlf_probe);
  P=NLF_PROBE; s=P.syst; ff0=s.ff;
  r.a=s.a; r.P=s.P; r.PAI00=s.PAI00; r.T1=cellfun(@full,s.T(1,:),'uni',0); r.T2=cellfun(@full,s.T(2,:),'uni',0);
  r.ss=cellfun(@full,s.steady_state,'uni',0); r.y=P.y; r.Q=full(s.Qfunc(s.a{1})); r.order_var=m.endogenous.name(m.order_var);
  algs={@switching_unscented_kalman_filter,@switching_cubature_kalman_filter}; nms={'ukf','ckf'};
  for k=1:2
    NLF_LOG={}; s2=s; s2.ff=@(varargin)nlf_logff(ff0,'ff',varargin{:});
    algs{k}(s2,P.y(:,1:4),P.U,P.z,P.options);
    X=[]; O=[]; R=[]; E=[];
    for i=1:numel(NLF_LOG), L=NLF_LOG{i}; X=[X L.in{2}]; O=[O L.out{1}]; R=[R L.in{1}]; E=[E L.in{3}(:)]; end
    r.([nms{k} '_X'])=X; r.([nms{k} '_O'])=O; r.([nms{k} '_R'])=R; r.([nms{k} '_E'])=E;
    [mf,ll,inc,rc]=filter(m,'data',db,'kf_user_algo',algs{k});
    r.([nms{k} '_ll'])=ll; r.([nms{k} '_inc'])=inc; r.([nms{k} '_rc'])=rc;
    [~,~,~,F]=algs{k}(s,P.y,P.U,P.z,P.options);
    r.([nms{k} '_PAItt'])=F.PAItt; r.([nms{k} '_PAI'])=F.PAI; r.([nms{k} '_PAItT'])=F.PAItT;
    r.([nms{k} '_att'])=F.att; r.([nms{k} '_a'])=F.a;
    r.([nms{k} '_upd_K'])=double(mf.Expected_updated_variables.K); r.([nms{k} '_upd_E'])=double(mf.Expected_updated_shocks.E);
    r.([nms{k} '_smo_K'])=double(mf.Expected_smoothed_variables.K);
  end
  out.(nm{1})=r;
end
save('../ref/nlf.mat','-struct','out','-v7'); disp('OK');
