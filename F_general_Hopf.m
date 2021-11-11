function [F] = F_general_Hopf(fn,X,phi,df)

alpha=X(1); % Parameter 
beta=X(2); % distance of the complex conjugate eigs from the real axis

dim=(length(X)-2)/3;
x=X(2+(1:dim)); % the variables from the model
Phi1=X(2+dim+(1:dim)); Phi2=X(2+2*dim+(1:dim)); % The real and imaginary parts of the eigenvector

%x=X(3:4); % the variables from the model
%Phi1=[X(5);X(6)]; Phi2=[X(7);X(8)]; % The real and imaginary parts of the eigenvector

f=feval(fn,x,alpha);
if nargin<4 || isempty(df)
    Df=finite_diff_fn(fn,x,alpha);
else
    Df=df(x,alpha);
end

Phi=Phi1+1i*Phi2;

tempF=Df*Phi-1i*beta*Phi;

F=[phi'*Phi1;phi'*Phi2-1;f;real(tempF);imag(tempF)];

end

