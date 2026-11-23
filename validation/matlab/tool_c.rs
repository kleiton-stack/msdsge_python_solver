endogenous x z
exogenous e u
parameters a sig
observables x
model
  z = a*z(-1) + sig*e;
  x = z + 0.1*u;
parameterization
  a, 0.9;
  sig, 0.2;
