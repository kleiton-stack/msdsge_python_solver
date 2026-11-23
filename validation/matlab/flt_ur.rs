endogenous x z w
exogenous e u v
parameters rho
observables x
model
  w = w(-1) + 0.1*v;
  z = rho*z(-1) + 0.3*e;
  x = z + w + 0.1*u;
parameterization
  rho, 0.8;
