function [loglik,Incr,retcode,Filters]=flt_probe(syst,y,U,z,options)
% User filter that only captures what dsge.filter hands over (initial state, covariance, etc.).
global NLF_PROBE
NLF_PROBE=struct('syst',syst,'y',y,'U',U,'z',z,'options',options);
loglik=0; Incr=zeros(size(y,2),1); retcode=0; Filters=[];
end
