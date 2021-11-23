function [X, alpha, beta, x, Phi1, Phi2]=newton_Hopf(fn, X, phi, varargin)

tol=1e-10;

F=F_general_Hopf(fn,X,phi,varargin);

display(['At the beggining, ||F(X)|| = ',num2str(norm(F))])
DF_inv = inv(finite_diff_general_Hopf(@F_general_Hopf,fn,X, phi));

k=0;
oldX = 20+X;
while (k<=200) && (norm(F)> tol)
    oldX = X;
    X = X - DF_inv*F_general_Hopf(fn,X,phi);
    DF_inv = inv(finite_diff_general_Hopf(@F_general_Hopf,fn,X, phi));
    F=F_general_Hopf(fn,X,phi,varargin);
    display(['||F(x)|| = ',num2str(norm(F))])
    display(['||x_n - x_(n+1)|| = ',num2str(norm(oldX-X))])
    k=k+1;
end

display(['||F(x)|| = ',num2str(norm(F)),', Newton iterations = ',num2str(k),', ||inv(DF)|| = ',num2str(norm(DF_inv))])
if norm(F)>tol || isnan(norm(F))
    error('Newton did not converge')
end


alpha=X(1); % Parameter 
beta=X(2); % distance of the complex conjugate eigs from the real axis

dim=(length(X)-2)/3;
x=X(2+(1:dim)); % the variables from the model
Phi1=X(2+dim+(1:dim)); Phi2=X(2+2*dim+(1:dim));

end
