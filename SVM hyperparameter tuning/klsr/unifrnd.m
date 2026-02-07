function r = unifrnd(a,b,varargin)
%UNIFRND fallback (no Statistics Toolbox).
% Usage: r = unifrnd(a,b,sz1,sz2,...) or r = unifrnd(a,b,sz)
if nargin < 2, error('unifrnd:NotEnoughInputs','Need a and b'); end
if isempty(varargin)
    sz = size(a);
    if isscalar(a) && ~isscalar(b), sz = size(b); end
    if isscalar(b) && ~isscalar(a), sz = size(a); end
    u = rand(sz);
else
    if numel(varargin)==1 && isnumeric(varargin{1}) && isvector(varargin{1})
        u = rand(varargin{1});
    else
        u = rand(varargin{:});
    end
end
r = a + (b-a).*u;
end
