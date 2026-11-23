// Validation case B: endogenous transition probabilities, unique steady state.
var X Z; varexo E; varobs X;
parameters sig gam;
markov_chain c(regimes=2);
parameters(c) rho;
sig = 0.3; gam = 1.5; rho(c,1) = 0.9; rho(c,2) = 0.2;
transition_probabilities(c);
  p(1,2) = 1/(1+exp(2-gam*Z));
  p(2,1) = 1/(1+exp(1+gam*Z));
end;
model;
  X = rho*X(-1) + sig*E;
  Z = 0.5*Z(+1) + 0.3*Z(-1) + X;
end;
