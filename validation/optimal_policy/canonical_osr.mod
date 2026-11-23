// Converted from Canonical_osr.rs (RISE) by msdsge.convert.rs_to_mod
var Y I PAI R DI;
varexo EI EPAI EY;
parameters beta_lag beta_lead beta_r gam_lag gam_y lamb_lag lamb_lead lamb_y sigi sigpai sigy;

beta_lag = 0.5000;
beta_lead = 0.4000;
beta_r = 0.9000;
gam_lag = 0.6000;
gam_y = 0.5000;
lamb_lag = 0.8000;
lamb_lead = 0.1000;
lamb_y = 0.3000;
sigi = 0.5000;
sigpai = 0.5000;
sigy = 0.5000;

model;
  Y=beta_lag*Y(-1)+beta_lead*Y(+1)-beta_r*R(-1)+sigy*EY;
  PAI=lamb_lag*PAI(-1)+lamb_lead*PAI(+1)+lamb_y*Y(-1)+sigpai*EPAI;
  I=gam_lag*I(-1)+(1-gam_lag)*(PAI(+4)+gam_y*Y)+sigi*EI;
  R=I-PAI(+1);
  DI=I-I(-1);
end;

planner_objective;
  -.5*(1*PAI^2+.3*Y^2+0.9*DI^2);
end;

estimated_params;
  gam_lag, 0.6000, 0.0000, 1.0000;
  gam_y, 0.5000, 0.0000, 10.0000;
end;
