// Converted from op_canG.rs (RISE) by msdsge.convert.rs_to_mod
var Y ZGDP ZI ZPAI ZY I PAI R;
varexo EGDP EI EPAI EY;
parameters beta_lag beta_lead beta_r gam_lag gam_y gyss lamb_lag lamb_lead lamb_y rhogdp rhoi rhopai rhoy vol_tp_1_2 vol_tp_2_1;
markov_chain vol(regimes=2);
parameters(vol) siggdp sigi sigpai sigy commit_prob;

gyss = 0;
beta_lag = 0.5000;
beta_lead = 0.4000;
beta_r = 0.9000;
gam_lag = 0.6000;
gam_y = 0.5000;
lamb_lag = 0.8000;
lamb_lead = 0.1000;
lamb_y = 0.3000;
rhogdp = 0.5000;
rhoi = 0.5000;
rhopai = 0.5000;
rhoy = 0.5000;
commit_prob(vol,1) = 0.3;
commit_prob(vol,2) = 0.8;
vol_tp_1_2 = 0.1000;
vol_tp_2_1 = 0.1000;
siggdp(vol,1) = 0.5000;
sigi(vol,1) = 0.5000;
sigpai(vol,1) = 0.5000;
sigy(vol,1) = 0.5000;
siggdp(vol,2) = 0.5000;
sigi(vol,2) = 0.5000;
sigpai(vol,2) = 0.5000;
sigy(vol,2) = 0.5000;

transition_probabilities(vol);
  p(1,2) = vol_tp_1_2;
  p(2,1) = vol_tp_2_1;
end;

model;
  Y=beta_lag*Y(-1)+beta_lead*Y(+1)-beta_r*R(-1)+ZY;
  PAI=lamb_lag*PAI(-1)+lamb_lead*PAI(+1)+lamb_y*Y(-1)+ZPAI;
  R=I-PAI(+1);
  ZI=rhoi*ZI(-1)+sigi*EI;
  ZPAI=rhopai*ZPAI(-1)+sigpai*EPAI;
  ZY=rhoy*ZY(-1)+sigy*EY;
  ZGDP=(1-rhogdp)*gyss+rhogdp*ZGDP(-1)+siggdp*EGDP;
end;

planner_objective(discount = 0.99, commitment=commit_prob);
  -.5*(1*PAI^2+.6*Y^2);
end;
