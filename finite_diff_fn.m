function [Dfn]=finite_diff_fn(fn, x, h, varargin)

if nargin<3 || isempty(h)
    h=1e-5;
end
M=length(x);
E=eye(M);
Dfn=zeros(M);
for j=1:M
    xh=x+h*E(:,j);
    fnxh=feval(fn,xh, varargin{:}); fnx=feval(fn,x,varargin{:});
    Dfn(:,j)=(fnxh-fnx)/h;
end
end