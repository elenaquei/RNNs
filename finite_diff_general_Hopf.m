function [DF]=finite_diff_general_Hopf(Fn,fn,X,Phi)

h=1e-5;
M=length(X);
E=eye(M);
DF=zeros(M);
for j=1:M
    Xh=X+h*E(:,j);
    FXh=feval(Fn,fn,Xh,Phi); FX=feval(Fn,fn,X,Phi);
    DF(:,j)=(FXh-FX)/h;
end
end