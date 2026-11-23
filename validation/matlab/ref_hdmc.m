% RISE historical decomposition with regime switching: it is Monte Carlo (the result changes on every call).
% Mean and standard deviation of 300 replications with 60 observations (about 5 minutes).
rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
here=pwd; work=tempname; mkdir(work); copyfile('flt_ms.rs',work); cd(work);
out=struct(); y=readmatrix([here '/../ref/filter_ms_data.csv']);
% historical decomposition with switching: it is Monte Carlo; mean and standard deviation of K replications with 60 observations
try
  db60=struct('x',ts('1990Q1',y(1:60)));
  m=rise('flt_ms'); m=solve(m); m=set(m,'data',db60,'kf_riccati_tol',0);
  K=300; Z=zeros(60,3,K); X=zeros(60,3,K);
  tic; for k=1:K, hd=historical_decomposition(m); Z(:,:,k)=double(hd.z); X(:,:,k)=double(hd.x); end; out.hdmc_time=toc;
  out.hdmc_z_mean=mean(Z,3); out.hdmc_z_std=std(Z,0,3); out.hdmc_x_mean=mean(X,3); out.hdmc_z_first=Z(:,:,1:5);
  f=filter(m); out.hdmc_smo=[double(f.Expected_smoothed_variables.x),double(f.Expected_smoothed_variables.z),double(f.Expected_smoothed_shocks.e),double(f.Expected_smoothed_shocks.u),double(f.smoothed_regime_probabilities.regime_1)];
catch err, out.hdmc_error=getReport(err); disp(getReport(err)); end
cd(here); save('../ref/hdmc.mat','-struct','out','-v7'); disp('OK');
