endogenous X Z
exogenous E
parameters rho sig c_tp_1_2 c_tp_2_1
parameters(c,2) mu
observables X
model
  X = mu*(1-rho) + rho*X(-1) + sig*E;
  Z = 0.5*Z(+1) + 0.3*Z(-1) + X;
parameterization
  rho, 0.8;
  sig, 0.3;
  c_tp_1_2, 0.1;
  c_tp_2_1, 0.2;
  mu(c,1), 1;
  mu(c,2), -0.5;
