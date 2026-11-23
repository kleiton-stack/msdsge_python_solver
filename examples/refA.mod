// Validation case A: regime-specific steady state, exogenous probabilities.
var X Z; varexo E; varobs X;
parameters rho sig;
markov_chain c(regimes=2);
parameters(c) mu;
rho = 0.8; sig = 0.3; mu(c,1) = 1; mu(c,2) = -0.5;
transition_probabilities(c); p(1,2) = 0.1; p(2,1) = 0.2; end;
model;
  X = mu*(1-rho) + rho*X(-1) + sig*E;
  Z = 0.5*Z(+1) + 0.3*Z(-1) + X;
end;
