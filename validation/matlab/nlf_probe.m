function [loglik,Incr,retcode,Filters]=nlf_probe(syst,y,U,z,options)
% User filter that only captures the inputs that dsge.filter passes to kf_user_algo.
global NLF_PROBE
NLF_PROBE=struct('syst',syst,'y',y,'U',U,'z',z,'options',options);
[loglik,Incr,retcode,Filters]=switching_unscented_kalman_filter(syst,y,U,z,options);
end
