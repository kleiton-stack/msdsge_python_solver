// Converted from examples/occbin/oneConstraint/occbin.rz (RISE) by msdsge.convert.rs_to_mod.
// Added by hand: calibration (baselineParameters.m and driver.m) and steady state (ssfile_full.m).
var Y R PI MC M C W MUZ N R_TAYLOR LAMBDA PIND DLOG_Y A RR;
varexo EA EZ ER EPI;
parameters sigma kappa kappa1 kappa2 epsilon phi eta beta rho_r rho_a pi_ss xi h gz d sig_pi sig_a sig_z sig_r r_zlb ocb_tp_1_2 ocb_tp_2_1;
markov_chain ocb(regimes=2);
parameters(ocb) bind;


transition_probabilities(ocb);
  p(1,2) = ocb_tp_1_2;
  p(2,1) = ocb_tp_2_1;
end;

model;
  R = bind*r_zlb + (1-bind)*R_TAYLOR;
  LAMBDA = A*(C - h*C(-1)/MUZ)^(-sigma);
  M = beta*d*LAMBDA(+1)/(MUZ(+1)*PI(+1)*LAMBDA);
  1/R = M;
  (epsilon/(epsilon - 1))*MC - 1*exp(sig_pi*EPI) - (phi/(epsilon-1))*PI*(PI - PIND) + (phi/(epsilon-1))*M*PI(+1)^2*(PI(+1) - PIND(+1))*MUZ(+1)*Y(+1)/Y = 0;
  R_TAYLOR = R_TAYLOR(-1)^rho_r*( steady_state(R_TAYLOR)*(PI/steady_state(PI))^kappa1*(Y/Y(-1))^kappa2 )^(1-rho_r)*exp(sig_r*ER);
  Y = C + (phi/2)*Y*(PI - PIND)^2;
  MC = W;
  MUZ = exp(log(steady_state(MUZ)) + sig_z*EZ);
  log(A) = rho_a*log(A(-1)) + sig_a*EA;
  W = kappa*N^eta/LAMBDA;
  W = MC*Y/N;
  PIND = PI(-1)^xi*steady_state(PI)^(1-xi);
  DLOG_Y = log(Y) - log(Y(-1)) + log(MUZ);
  RR = R/PI(+1);
end;

occbin_constraints;
  R>=1;
end;

// calibration (baselineParameters.m, driver.m); kappa comes from ssfile_full.m
h = 0.8; gz = 0.001; pi_ss = 1.005; beta = 0.9975; kappa1 = 5; kappa2 = 0.5; epsilon = 6; phi = 10;
eta = 2; xi = 0.5; rho_r = 0.8; rho_a = 0.9; sig_r = 0.01; sig_a = 0.01; sig_pi = 0.01; sig_z = 0.01;
sigma = 2; d = 1; r_zlb = 1; ocb_tp_1_2 = 0; ocb_tp_2_1 = 0;
bind(ocb,1) = 0; bind(ocb,2) = 1;
kappa = (1 - h/exp(gz))^(-sigma)*(epsilon - 1)/epsilon;

steady_state_model;
  PI = pi_ss;
  MUZ = exp(gz);
  R_TAYLOR = pi_ss*exp(gz)/beta;
  R = R_TAYLOR;
  A = 1;
  Y = 1;
  C = 1;
  LAMBDA = (C - h*C/MUZ)^(-sigma);
  PIND = PI;
  M = 1/R;
  MC = (epsilon - 1)/epsilon;
  N = 1;
  W = MC;
  DLOG_Y = log(MUZ);
  RR = R/PI;
end;
