rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
out=struct();
names={'ar','ms'};
for k=1:2
  y=readmatrix(['../ref/filter_' names{k} '_data.csv']);
  db=struct('x',ts('1990Q1',y));
  m=rise(['flt_' names{k}]);
  m=solve(m);
  [mf,LogLik,Incr,retcode]=filter(m,'data',db);
  r=struct('LogLik',LogLik,'Incr',Incr,'retcode',retcode);
  r.cls=class(mf);
  if isstruct(mf), f=mf; else, f=mf.filtering; end
  r.top=fieldnames(f);
  fn=r.top;
  for q=1:numel(fn)
    v=f.(fn{q});
    if isstruct(v)
      r.(['sub_' fn{q}])=fieldnames(v);
      if contains(fn{q},'regime_prob')
        g=fieldnames(v);
        for w=1:numel(g), r.([fn{q} '__' g{w}])=double(v.(g{w})); end
      end
    end
  end
  out.(names{k})=r;
end
save('../ref/filter.mat','-struct','out','-v7'); disp('OK');
