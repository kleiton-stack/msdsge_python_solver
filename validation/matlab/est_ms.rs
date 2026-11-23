endogenous x z
exogenous e u
parameters c_tp_1_2 c_tp_2_1 su
parameters(c,2) a sig
observables x
model
  z = a*z(-1) + sig*e;
  x = z + su*u;
parameterization
  c_tp_1_2, 0.05, 0.02, 0.20, beta_pdf(.90);
  c_tp_2_1, 0.10, 0.02, 0.30, beta_pdf(.90);
  su, 0.1, 0.05, 0.30, inv_gamma_pdf(.90);
  a(c,1), 0.9, 0.5, 0.95, beta_pdf(.90);
  a(c,2), 0.1, -0.5, 0.5, normal_pdf(.90);
  sig(c,1), 0.2, 0.05, 1.0, gamma_pdf(.90);
  sig(c,2), 1.5, 0.5, 3.0, weibull_pdf(.90);
