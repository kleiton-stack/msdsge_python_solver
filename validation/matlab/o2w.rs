endogenous C K A
exogenous E
parameters alpha beta sig delta abar
parameters(s,2) rho
model
  1/C = beta/C(+1)*(alpha*A(+1)*K^(alpha-1) + 1 - delta);
  K = A*K(-1)^alpha + (1-delta)*K(-1) - C;
  log(A) = (1-rho)*log(abar) + rho*log(A(-1)) + sig*E;
  ! s_tp_1_2 = 0.1/(0.1+0.9*exp(2*(A-1)+0.5*(C-3)));
  ! s_tp_2_1 = 0.3/(0.3+0.7*exp(-2*(A-1)+0.25*(C-3)));
steady_state_model
  A = abar;
  K = ((1/beta - 1 + delta)/(alpha*abar))^(1/(alpha-1));
  C = abar*K^alpha - delta*K;
parameterization
  alpha, 0.33;
  beta, 0.99;
  sig, 0.02;
  delta, 0.025;
  abar, 1.3;
  rho(s,1), 0.9;
  rho(s,2), 0.5;
