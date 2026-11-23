function varargout=nlf_logff(f,tag,varargin)
% Wraps a RISE handle and logs the inputs and outputs of each call (numerical exploration).
global NLF_LOG
[varargout{1:max(nargout,1)}]=f(varargin{:});
NLF_LOG{end+1}=struct('tag',tag,'in',{varargin},'out',{varargout});
end
