function [solutions, positive_lyap_index, negative_lyap_index, positive_lyap, negative_lyap, unproven] = totallygeneral_RHS_Hopf(dim, allders, validation)
% find Hopf bifurcation in the generic system defined by the composition of
% an elemntwise function "nonlin_struct" and a function "lin_struct".
% The naming convention comes from the RNNs defiuned by sigma \compose W(t)
% This code refers to the paper "PLACEHOLDER" by E. Queirolo and C. Kuehn
%
% function asym_RHS_Hopf(dim, nonlin_struct, lin_struct)
%
% INPUTS
% dim           integer, dimension of the system
% nonlin_struct, lin_struct functions, called without arguments they return
%               their derivatives
% validation    bool, if results are validated (DEFAULT = 1)
% all_ders() returns
%     @(x, par) f(x, par) returning a vector of length dim
%     @(x, par) d_x_f(x, par) returning a square matrix of size (dim, dim)
%     @(x, par) d_par_f(x, par) returning a vector of length dim
%     @(x, par) d_parpar_f(x, par) returning a vector of length dim
%     @(x, par) d_xpar_f(x, par) returning a square matrix of size (dim, dim)
%     @(x, par, y) d_xx_fv(x, par) returning a square matrix of size (dim, dim)
%     @(x, par, y, z) d_xxx_fvw(x, par) returning a square matrix of size (dim, dim)
% where x, y, z are vectors of length dim and par is a parameter of size 1
% and y and z are the directions for the directional derivatives.
%
% OUTPUTS
%
% solutions     each row is a numerical solution, stored as 
%               par, eigenval_imag, x, eigenvec_real, eigenvec_imag
% positive_lyap_index, negative_lyap_index, positive_lyap, negative_lyap, unproven

if nargin < 4 || isempty(validation)
    validation = 1;
end


% definition of the full matrix 
out = allders();
f = out{2};
d_x_f = out{3};
d_par_f = out{4};  
d_xx_fv = out{5}; 
d_xpar_f = out{6};
d_xxx_fvw = out{7};

phi = randi([-5,5],[dim,1]);
phi = phi/norm(phi);

% tests
% x =rand(dim,1);
% a = 89;
% size(f(x,a))
% size(df(x,a))
% size(dparf(x,a))
% size(dxparf(x,a))
% size(dxxfv(x,a, x))
% R1 = dxxfv(x,a, x);
% return
% checked w.r.t. asym_RHS_Hopf, all correct


% full Hopf problem - wrappers 
F = @(X) wrapper_Hopf(X, dim, f, d_x_f, phi);
DF = @(X) wrapper_DHopf(X, dim, d_x_f, d_xx_fv, d_par_f, d_xpar_f, phi);

X_merge = @(par, eigenval_imag, x, eigenvec_real, eigenvec_imag) [par; eigenval_imag; x; eigenvec_real; eigenvec_imag];

% Numerical search for a Hopf bifurcation
solutions = [];
poor_solutions = [];


% equilibrium for the ODE
par = 0;
x = zeros(dim,1); 
x = Newton(@(x)f(x,par), @(x)d_x_f(x,par), x);

% at the found equilibrium, find the best eigenpair
[V,L] = eigs(d_x_f(x, par), dim);

% loop through eigenpairs to find associated Hopf bifurcations
for i = 1:dim
    eigenval_imag = imag(L(i,i));
    
    % rotate the eigenvector to fit the linear scaling
    first_eig = V(:,i);
    scaling = 1i / (phi.' * first_eig);
    first_eig = scaling * first_eig;
    eigenvec_real = real(first_eig);
    eigenvec_imag = imag(first_eig);
    
    X = X_merge(par, eigenval_imag, x, eigenvec_real, eigenvec_imag);
    
    % Newton to find Hopf bifurcation
    try
        [X]=Newton(F, DF, X);
        if log(norm(X))<10
            if size(solutions,1)<1 || min(abs(solutions(:,1)-X(1)))>10^-7
                solutions(end+1,:) = X;
                % fprintf('A Hopf bifurcation has been found\n')
            end 
        else
            % fprintf('Newton converged but the norm is too large to be trusted, norm = %e\n', norm(X))
            poor_solutions(end+1,:) = X;
        end
    catch
        % error('No Hopf bifurcation found')
    end
    
end
fprintf('Finished the numerical search, %i numerical solutions found out of %i expected.\n', size(solutions,1), floor(dim/2)) 
if validation
 fprintf('Starting the validation now.\n')
end
% set up for validation loop - storage 
number_of_positive_stab = 0;
number_of_negative_stab = 0;
not_proven = 0;

positive_lyap_index = ([]);
negative_lyap_index = ([]);
positive_lyap = intval([]);
negative_lyap = intval([]);
unproven = [];

% set up for validation proof - problem dependent derivatives
bool_val = 1;

% % TESTS
% x =rand(dim,1);
% y =rand(dim,1);
% v =rand(dim,1);
% a = 89;
% size(d_xxx_fvw(x,a, y, v))
% R1 = d_xxx_fvw(x,a, y, v);
% return

% already defined - d_par_f = @(x,a) der_par_RHS(x, W(a), d_par_W(a));
% already defined - d_xpar_f = @(x, a) der_par_xRHS(x, W(a), d_par_W(a));
% dparxxf = @(x, a, v) der_x_x_par_RHS(x, W(a), d_par_W(a), v);
% dparparf = @(x, a)der_par_par_RHS(x, W(a), d_par_W(a), d_parpar_W(a));
% dparparxf = @(x, a, v) der_x_par_par_RHS(x, W(a), d_par_W(a), d_parpar_W(a));

% validation loop
for i = 1: size(solutions,1)*(validation~=0)
    solution = solutions(i,:);
    X = solution.';
    
    try
        % Validation - including computation of first Lyapunov coeff
        
        % % more precise, but needs more derivatives
        % [x_star,lambda_star,eigenvec,eigenval, l1] = ...
        %    algebraic_hopf(f,df,ddfv,dddfvw, dparf, dxparf,dparxxf, dparparf,dparparxf, dim,X,phi,bool_val);
        % % easier, basically same precision
        [x_star,lambda_star,eigenvec,eigenval, l1] = ...
            algebraic_hopf_simple(f,d_x_f,d_xx_fv,d_xxx_fvw,d_par_f, d_xpar_f,...
            dim,X,phi,bool_val);
        
        if l1>0
            number_of_positive_stab = number_of_positive_stab+1;
            positive_lyap_index(end+1) = i;
            positive_lyap(end+1) = l1;
        else
            number_of_negative_stab = number_of_negative_stab+1;
            negative_lyap_index(end+1)= i;
            negative_lyap(end+1)= l1;
            fprintf('Negative stability found!')
        end
    catch
        not_proven = not_proven+1;
        unproven(end+1) = i;
    end
    if dim >= 10 && mod(i,5)==0
        fprintf('Validated %i Hopf bifurcations out of %i.\n', i, size(solutions,1))
    end
end
if ~isempty(unproven)
    fprintf('We could not prove %i bifurcations\n',length(unproven));
elseif validation
    fprintf('We could validate all Hopf bifurcations\n');
end

end

% all needed functionalities

function [par, eigenval_imag, x, eigenvec_real, eigenvec_imag] = split_into_elements(X, dim)
if length(X) ~= 3*dim+2
    error('Sizes are not compatible')
end

par = X(1);
eigenval_imag = X(2);
x = X(2+(1:dim));
eigenvec_real = X(dim+3:2*dim+2);
eigenvec_imag = X(2*dim+3:3*dim+2);
end


function y = wrapper_Hopf(X, dim, f, df, phi)
[par, eigenval_imag, x, eigenvec_real, eigenvec_imag] = split_into_elements(X, dim);
y = f_Hopf(f, df, phi, par, eigenval_imag, x, eigenvec_real, eigenvec_imag);
end

function DF = wrapper_DHopf(X, dim, df, ddf, dparf, dxparf, phi)
[par, eigenval_imag, x, eigenvec_real, eigenvec_imag] = split_into_elements(X, dim);
DF = Df_Hopf([], df, ddf, dparf, dxparf, phi, par, eigenval_imag, x, eigenvec_real, eigenvec_imag);
end


% function DF = Df_Hopf(~, df, ddf, dalphaf, dxalphaf, phi, alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag)
% % Hopf problem
% %
% % f(x, alpha) = 0
% % Df*Phi-1i*beta*Phi = 0
% % 
% % beta eigenvalue
% % Phi eigenvector
% 
% v1 = eigenvec_real;
% v2 = eigenvec_imag;
% 
% Dxf_x = feval(df, x, alpha);
% Dxxf_x_v1 = tensor_prod(feval(ddf, x, alpha),v1);
% Dxxf_x_v2 = tensor_prod(feval(ddf, x, alpha),v2);
% Dalphaf_x = feval(dalphaf, x, alpha);
% Dxalpha_x = feval(dxalphaf, x, alpha);
% 
% DF = [0,                0,                      zeros(1,length(x)),     phi',                           zeros(1,length(x))
%       0,                0,                      zeros(1,length(x)),     zeros(1,length(x)),             phi'
%       Dalphaf_x,        zeros(length(x),1)      Dxf_x,                  zeros(length(x)),               zeros(length(x))           
%       Dxalpha_x * v1,   v2,                     Dxxf_x_v1,              Dxf_x,                          eigenval_imag*eye(length(x))  
%       Dxalpha_x * v2,   -v1,                    Dxxf_x_v2,              -eigenval_imag*eye(length(x)),  Dxf_x];
% 
% return
% 
% end
% 
% 
% function Av = tensor_prod(A,v)
% dims = size(A);
% if length(dims)~=3
%     error('not coded')
% end
% Av = zeros(dims(1:end-1));
% for i =1:size(A,1)
%     for j = 1:size(A,2)
%         Av(i,j) = sum(squeeze(A(i,j,:)).*v);
%     end
% end
% 
% end