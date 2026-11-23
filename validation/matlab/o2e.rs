endogenous C K A
exogenous E
parameters alpha beta rho sig
parameters(s,2) delta
model
  1/C = beta/C(+1)*(alpha*A(+1)*K^(alpha-1) + 1 - delta);
  K = A*K(-1)^alpha + (1-delta)*K(-1) - C;
  log(A) = rho*log(A(-1)) + sig*E;
  ! s_tp_1_2 = 0.1/(0.1+0.9*exp(2*(A-1)+0.02*(K-20)));
  ! s_tp_2_1 = 0.3/(0.3+0.7*exp(-2*(A-1)+0.01*(K-20)));
steady_state_model
  A = 1;
  K = ((1/beta - 1 + delta)/alpha)^(1/(alpha-1));
  C = K^alpha - delta*K;
parameterization
  alpha, 0.33;
  beta, 0.99;
  rho, 0.9;
  sig, 0.02;
  delta(s,1), 0.025;
  delta(s,2), 0.05;
