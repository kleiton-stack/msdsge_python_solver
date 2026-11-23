// Farmer, Waggoner and Zha (2011, JEDC). Parameterisation 1 (unique solution).
// Source: RISE examples/MarkovSwitching/FarmerWaggonerZha2010/fwz10_nasty.rs
var PAI R;
varexo E;
markov_chain a(regimes=2);
parameters(a) betta delta phi rho;

betta = 1; delta = 0; rho = 0.9;
phi(a,1) = 0.5; phi(a,2) = 0.8;

transition_probabilities(a);
  p(1,2) = 1-0.8;
  p(2,1) = 1-0.9;
end;

model;
  phi*PAI = PAI(+1) + delta*PAI(-1) + betta*R;
  R = rho*R(-1) + E;
end;
