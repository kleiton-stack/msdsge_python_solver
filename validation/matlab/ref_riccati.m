% Riccati shortcut of the RISE linear filter: increments for a grid of kf_riccati_tol.
% Used to identify the stopping rule by numerical comparison (validation/explore_riccati.py).
rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
out=struct();
names={'ms','ar'};
for k=1:2
  y=readmatrix(['../ref/filter_' names{k} '_data.csv']); db=struct('x',ts('1990Q1',y));
  m=rise(['flt_' names{k}]); m=solve(m);
  tols=[0 10.^(-12:0.25:0)]; Incr=zeros(numel(y),numel(tols)); LL=zeros(1,numel(tols));
  for i=1:numel(tols)
    [~,LL(i),inc]=filter(m,'data',db,'kf_riccati_tol',tols(i)); Incr(:,i)=inc(:);
  end
  out.(names{k})=struct('tols',tols,'Incr',Incr,'LogLik',LL);
end
save('../ref/riccati.mat','-struct','out','-v7'); disp('OK');
