function DF = Df_Hopf(~, df, ddf, dalphaf, dxalphaf, phi, alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag)
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
try
    Dxxf_x_v1 = feval(ddf, x, alpha, v1);
    Dxxf_x_v2 = feval(ddf, x, alpha, v2);
catch
    Dxxf_x_v1 = tensor_prod(feval(ddf, x, alpha),v1);
    Dxxf_x_v2 = tensor_prod(feval(ddf, x, alpha),v2);
end

Dalphaf_x = feval(dalphaf, x, alpha);
Dxalpha_x = feval(dxalphaf, x, alpha);

DF = [0,                0,                      zeros(1,length(x)),     phi',                           zeros(1,length(x))
      0,                0,                      zeros(1,length(x)),     zeros(1,length(x)),             phi'
      Dalphaf_x,        zeros(length(x),1)      Dxf_x,                  zeros(length(x)),               zeros(length(x))           
      Dxalpha_x * v1,   v2,                     Dxxf_x_v1,              Dxf_x,                          eigenval_imag*eye(length(x))  
      Dxalpha_x * v2,   -v1,                    Dxxf_x_v2,              -eigenval_imag*eye(length(x)),  Dxf_x];

return
end

