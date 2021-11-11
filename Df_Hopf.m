function DF = Df_Hopf(~, df, ddf, dalphaf, dxalphaf, phi, x, alpha, eigenvec_real, eigenvec_imag, eigenval_imag)
% Hopf problem
%
% f(x, alpha) = 0
% Df*Phi-1i*beta*Phi = 0
% 
% beta eigenvalue
% Phi eigenvector

v1 = eigenvec_real;
v2 = eigenvec_imag;

Dxf_x = feval(df, x, alpha);
Dxxf_x_v1 = feval(ddf, x, alpha, v1);
Dxxf_x_v2 = feval(ddf, x, alpha, v2);
Dalphaf_x = feval(dalphaf, x, alpha);
Dxalpha_x = feval(dxalphaf, x, alpha);

DF = [zeros(1,length(x)), 0,                phi',                           zeros(1,length(x)),             0;
      zeros(1,length(x)), 0,                zeros(1,length(x)),             phi',                           0;
      Dxf_x,              Dalphaf_x,        zeros(length(x)),               zeros(length(x)),               zeros(length(x),1)
      Dxxf_x_v1,          Dxalpha_x * v1,   Dxf_x,                          eigenval_imag*eye(length(x)),   v2
      Dxxf_x_v2,          Dxalpha_x * v2,   -eigenval_imag*eye(length(x)),  Dxf_x,                          -v1];

return