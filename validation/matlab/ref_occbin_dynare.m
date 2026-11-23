% Cross-check against OccBin in Dynare 6.4 (Guerrieri and Iacoviello, 2015) on the model in RISE's
% examples/occbin/oneConstraint, with the same shocks as the RISE simulation (ref_occbin.m).
addpath('/Applications/Dynare/6.4-arm64/matlab');
here=pwd; r=load(fullfile(here,'..','ref','occbin.mat'));
exo={'EA','EZ','ER','EPI'};                       % varexo order in the .mod
[~,loc]=ismember(exo,cellstr(r.one.exo));
global occbin_shocks__
occbin_shocks__=r.one.shocks(:,loc);
w=tempname; mkdir(w); copyfile(fullfile(here,'occbin_zlb_dynare.mod'),w); cd(w);
dynare occbin_zlb_dynare noclearall nolog
out=struct();
out.endo=cellstr(M_.endo_names); out.exo=exo; out.shocks=occbin_shocks__;
out.piecewise=oo_.occbin.simul.piecewise(1:size(occbin_shocks__,1),:);   % deviations from the steady state
out.linear=oo_.occbin.simul.linear(1:size(occbin_shocks__,1),:);
out.ss=oo_.steady_state;
rh=oo_.occbin.simul.regime_history; reg=zeros(numel(rh),1);
for t=1:numel(rh), reg(t)=rh(t).regime(1); end
out.regime=reg;                                    % 1 if the constraint binds at t
out.dynare_version=dynare_version;
cd(here);
save('../ref/occbin_dynare.mat','-struct','out','-v7'); disp('OK');
