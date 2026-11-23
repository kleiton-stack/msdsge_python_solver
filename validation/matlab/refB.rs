endogenous X Z
exogenous E
parameters sig gam
parameters(c,2) rho
observables X
model
  X = rho*X(-1) + sig*E;
  Z = 0.5*Z(+1) + 0.3*Z(-1) + X;
  ! c_tp_1_2 = 1/(1+exp(2-gam*Z));
  ! c_tp_2_1 = 1/(1+exp(1+gam*Z));
parameterization
  sig, 0.3;
  gam, 1.5;
  rho(c,1), 0.9;
  rho(c,2), 0.2;
