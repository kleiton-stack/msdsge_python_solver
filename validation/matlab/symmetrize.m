function A=symmetrize(A)
% Patch: diffuse_lyapunov_equation (RISE 1b2edfa) calls symmetrize without the package prefix; the function exists in utils.cov.
A=utils.cov.symmetrize(A);
end
