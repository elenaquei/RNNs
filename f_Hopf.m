function y = f_Hopf(f, df, phi, alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag)
% Hopf problem
%
% f(x, alpha) = 0
% Df*Phi-1i*beta*Phi = 0
% 
% beta eigenvalue
% Phi eigenvector

f_x = feval(f, x, alpha);

% real and imaginary equations (computed already separate)
eq1 = feval(df, x, alpha) * eigenvec_real + eigenval_imag * eigenvec_imag;
eq2 = feval(df, x, alpha) * eigenvec_imag - eigenval_imag * eigenvec_real;

y = [phi'*eigenvec_real;phi'*eigenvec_imag-1;f_x; eq1; eq2];

return