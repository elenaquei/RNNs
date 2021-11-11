function DDF= second_derivative_F(X,~,df)
% function DDF= second_derivative_F(X,phi,df)
%
% DDF \in R^length(X)

alpha=X(1); % Parameter 
beta=X(2); % distance of the complex conjugate eigs from the real axis

dim=(length(X)-2)/3;
x=X(2+(1:dim)); % the variables from the model
v1=X(2+dim+(1:dim)); v2=X(2+2*dim+(1:dim)); % The real and imaginary parts of the eigenvector

one=ones(dim,1);%[1;1];

[DxFn,Dalphafn,DxxFnV1,DalphaxFn,DalphaxxFnV1,DalphaalphaFn,...
    DalphaalphaxFn,DxxxFnV1] = df(x,alpha,v1);
[~,~,DxxFnV2,~,DalphaxxFnV2,~,~,DxxxFnV2] = df(x,alpha,v2);

[~,~,DxxFnONE,~,DalphaxxFnONE,~,~,DxxxFnONE] = df(x,alpha,one);

DDF=[0
    0
    DxxFnONE*one+2*DalphaxFn*one+DalphaalphaFn
    DxxxFnV1*one+2*DxxFnONE*one+2*DalphaxxFnV1*one+DalphaalphaxFn*v1+2*DalphaxFn*one+2*one
    DxxxFnV2*one+2*DxxFnONE*one+2*DalphaxxFnV2*one+DalphaalphaxFn*v2+2*DalphaxFn*one-2*one];

if length(X)~=length(DDF)
    error('dimensional problem')
end