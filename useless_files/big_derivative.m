function DF = big_derivative(df,X, phi)
% function DF = big_derivative(df,X, phi)
%
% INPUT
% df       inline function, takes as input (x, alpha, v1) and returns
% (DxFn, DalphaFn, DxxFnV1, DalphaxFn)
% X 

alpha=X(1); % Parameter 
beta=X(2); % distance of the complex conjugate eigs from the real axis


dim=(length(X)-2)/3;
n = dim;
x=X(2+(1:dim)); % the variables from the model
v1=X(2+dim+(1:dim)); v2=X(2+2*dim+(1:dim)); % The real and imaginary parts of the eigenvector


if size(v1,1)==1 
    v1=v1.';
end
if size(v2,1)==1 
    v2=v2.';
end
[DxFn,Dalphafn,DxxFnV1,DalphaxFn] = df(x,alpha,v1);
[~,~,DxxFnV2,~] = df(x,alpha,v2);

DF=[zeros(1,n)  0              0              phi'            zeros(1,n)
    zeros(1,n)  0              0              zeros(1,n)      phi'
    DxFn        Dalphafn       zeros(n,1)     zeros(n,n)      zeros(n,n)
    DxxFnV1     DalphaxFn*v1   v2             DxFn            beta*eye(n,n)
    DxxFnV2     DalphaxFn*v2   -v1            -beta*eye(n,n)  DxFn];
    