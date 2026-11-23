// 3-equation New Keynesian model with a regime-switching interest-rate rule.
// Regime 1: active policy. Regime 2: passive. Endogenous 2->1 transition.
var y pi i;
varexo e_m;
parameters beta kappa rho;

markov_chain pol(regimes=2);
parameters(pol) phi_pi;

beta = 0.99; kappa = 0.1; rho = 0.8;
phi_pi(pol,1) = 1.5;
phi_pi(pol,2) = 0.9;

transition_probabilities(pol);
  p(1,2) = 0.05;
  p(2,1) = 1/(1+exp(2-2*pi(-1)));
end;

model;
  y = y(+1) - (i - pi(+1));
  pi = beta*pi(+1) + kappa*y;
  i = rho*i(-1) + (1-rho)*phi_pi*pi + e_m;
end;

shocks;
  var e_m; stderr 0.25;
end;
