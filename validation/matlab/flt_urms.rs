endogenous x z w
exogenous e u v
parameters c_tp_1_2 c_tp_2_1
parameters(c,2) a sig
observables x
model
  w = w(-1) + 0.1*v;
  z = a*z(-1) + sig*e;
  x = z + w + 0.1*u;
parameterization
  c_tp_1_2, 0.05;
  c_tp_2_1, 0.10;
  a(c,1), 0.9;
  a(c,2), 0.1;
  sig(c,1), 0.2;
  sig(c,2), 1.5;
