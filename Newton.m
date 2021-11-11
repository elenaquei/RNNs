function x_out = Newton(F, DF, x_in,varargin)

tol=1e-10;

%display(['At the beggining, ||F(X)|| = ',num2str(norm(F(x_in)))])
x = x_in;
k=0;

while (k<=200) && (norm(F(x))> tol)
    oldX = x;
    DF_x = DF(x);
    x = x - DF_x\F(x);
    if any(isnan(x))
        error('Newton diverged to NaN territory')
    end
    % display(['||F(x)|| = ',num2str(norm(F(x)))])
    % display(['||x_n - x_(n+1)|| = ',num2str(norm(oldX-x))])
    k=k+1;
end
DF_inv = inv(DF(x_in));
% display(['||F(x)|| = ',num2str(norm(F(x))),', Newton iterations = ',num2str(k),', ||inv(DF)|| = ',num2str(norm(DF_inv))])
x_out = x;
if any(isnan(F(x))) || norm(F(x))>tol
    error('Newton did not converge')
end


end
