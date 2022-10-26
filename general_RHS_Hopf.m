function [solutions, positive_lyap_index, negative_lyap_index, positive_lyap, negative_lyap, unproven] = general_RHS_Hopf(dim, nonlin_struct, lin_struct, validation)
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
% nonlin_struct() returns elementwise functions
%     @(y) nonlin(y)    
%     @(y) d_nonlin(y)
%     @(y) dd_nonlin(y)
%     @(y) ddd_nonlin(y)
% lin_struct() returns
%     @(x, par) lin(x, par) returning a vector of length dim
%     @(x, par) d_x_lin(x, par) returning a square matrix of size (dim, dim)
%     @(x, par) d_par_lin(x, par) returning a vector of length dim
%     @(x, par) d_parpar_lin(x, par) returning a vector of length dim
%     @(x, par) d_xpar_lin(x, par) returning a square matrix of size (dim, dim)
%     @(x, par, y) d_xx_lin(x, par) returning a square matrix of size (dim, dim)
%     @(x, par, y, z) d_xxx_lin(x, par) returning a square matrix of size (dim, dim)
% where x, y, z are vectors of length dim and par is a parameter of size 1
% and y and z are the directions for the directional derivatives.
%
% OUTPUTS
%
% solutions     each column is a numerical solution, stored as 
%               par, eigenval_imag, x, eigenvec_real, eigenvec_imag
% positive_lyap_index, negative_lyap_index, positive_lyap, negative_lyap, unproven

if nargin < 4 || isempty(validation)
    validation = 1;
end


% definition of the full matrix 
[W, d_x_W, d_par_W, d_parpar_W, d_xpar_W, d_xx_W, d_xxx_W] = lin_struct();

[nonlin, d_nonlin, dd_nonlin, ddd_nonlin] = nonlin_struct();

phi = randi([-5,5],[dim,1]);
phi = phi/norm(phi);

% construction of the Hopf problem based on the just defined elements:
f = @(x, par) rhs(x, par, nonlin, W); % DONE
df = @(x, par) der_RHS(x, par, nonlin, d_nonlin, W, d_x_W); % DONE
d_par_f = @(x, par) der_par_RHS(x, par, nonlin, d_nonlin, W, d_par_W); % DONE
d_xpar_f = @(x, par) der_par_xRHS(x, par, nonlin, d_nonlin, dd_nonlin, W, d_x_W, d_par_W, d_xpar_W); % DONE
dxxfv = @(x, par, dir) dir_der2RHS(x, par, dir, nonlin, d_nonlin, dd_nonlin, W, d_x_W, d_xx_W); % DONE

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
F = @(X) wrapper_Hopf(X, dim, f, df, phi);
DF = @(X) wrapper_DHopf(X, dim, df, dxxfv, d_par_f, d_xpar_f, phi);

X_merge = @(par, eigenval_imag, x, eigenvec_real, eigenvec_imag) [par; eigenval_imag; x; eigenvec_real; eigenvec_imag];

% Numerical search for a Hopf bifurcation
solutions = [];
poor_solutions = [];


% equilibrium for the ODE
par = 0;
x = zeros(dim,1); 
x = Newton(@(x)f(x,par), @(x)df(x,par), x);

% at the found equilibrium, find the best eigenpair
[V,L] = eigs(df(x, par), dim);

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

d_xxx_fvw = @(x, par, dir1, dir2) dir_der3RHS(x, par, dir1, dir2, nonlin, d_nonlin, ...
    dd_nonlin, ddd_nonlin, W, d_x_W, d_xx_W, d_xxx_W);

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
            algebraic_hopf_simple(f,df,dxxfv,d_xxx_fvw,d_par_f, d_xpar_f,...
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


function x_dot = rhs(x, par, nonlin, W)
% INPUTS
% x             vector
% par           float, parameter
% nonlin        nonlinear function, takes a vector and returns a vector
% W             linear function, takes a vector and returns a vector

x_dot = nonlin(W(x, par));
end

function der = der_RHS(x, par, nonlin, d_nonlin, W, d_W)
% INPUTS
% x             vector
% par           float, parameter
% nonlin        nonlinear function, takes a vector and returns a vector
% d_nonlin      nonlinear function, takes a vector and returns a vector
% W             linear function, takes a vector and returns a vector
% d_W           linear function, takes a vector and returns a matrix
% OUTPUT
% der           matrix

der = diag(d_nonlin(W(x, par))) * d_W(x, par);
end

function dparRHS =  der_par_RHS(x, par, nonlin, d_nonlin, W, d_par_W)
% INPUTS
% x             vector
% par           float, parameter
% nonlin        nonlinear function, takes a vector and returns a vector
% d_nonlin      nonlinear function, takes a vector and returns a vector
% W             linear function, takes a vector and returns a vector
% d_par_W       linear function, takes a vector and returns a vector
% OUTPUT
% der           vector


dparRHS = diag(d_nonlin(W(x, par))) * d_par_W(x, par);
end


function dparxRHS = der_par_xRHS(x, par, nonlin, d_nonlin, dd_nonlin, W, d_W, d_par_W, d_xpar_W)
% INPUTS
% x             vector
% par           float, parameter
% nonlin        nonlinear function, takes a vector and returns a vector
% d_nonlin      nonlinear function, takes anything and returns the same shape
% dd_nonlin     nonlinear function, takes anything and returns the same shape
% W             linear function, takes a vector and returns a vector
% d_x_W         linear function, takes a vector and returns a matrix
% d_par_W       linear function, takes a vector and returns a vector
% d_xpar_W      linear function, takes a vector and returns a matrix
% OUTPUT
% der           matrix

v = d_par_W(x, par);

y = W(x, par);

dparxRHS = diag(  dd_nonlin(y) .* v) * d_W(x,par) + diag( d_nonlin(y) ) * d_xpar_W(x, par);

end



function dxxfv = dir_der2RHS(x, par, dir, nonlin, d_nonlin, dd_nonlin, W, d_W, d_xx_W)
% INPUTS
% x             vector
% par           float, parameter
% dir           vector, direction
% nonlin        nonlinear function, takes a vector and returns a vector
% d_nonlin      nonlinear function, takes anything and returns the same shape
% dd_nonlin     nonlinear function, takes anything and returns the same shape
% W             linear function, takes a vector and returns a vector
% d_x_W         linear function, takes a vector and returns a matrix
% d_xx_W        linear function, takes two vector and returns a matrix
% OUTPUT
% der           matrix
y = W(x,par);

dxxfv = diag( dd_nonlin(y) .* (d_W(x, par)* dir)) * d_W(x, par) + diag( d_nonlin(y) ) * d_xx_W(x, par, dir);

end


function dxxxfvw = dir_der3RHS(x, par, dir1, dir2, nonlin, d_nonlin, d2_nonlin, d3_nonlin, W, d_W, d_xx_W, d_xxx_W)
% INPUTS
% x             vector
% par           float, parameter
% dir1, dir2    vectors, directions
% nonlin        nonlinear function, takes a vector and returns a vector
% d_nonlin      nonlinear function, takes anything and returns the same shape
% d2_nonlin     nonlinear function, takes anything and returns the same shape
% d3_nonlin     nonlinear function, takes anything and returns the same shape
% W             linear function, takes a vector and par and returns a vector
% d_x_W         linear function, takes a vector and par and returns a matrix
% d_xx_W        linear function, takes a vector, par and direction and returns a matrix
% d_xxx_W       linear function, takes a vector, par and two directions and returns a matrix
% OUTPUT
% der           matrix

y = W(x, par);

dxxxfvw = diag(d3_nonlin(y).* (d_W(x,par)*dir1).*(d_W(x,par)*dir2)) * d_W(x,par) + ...
    diag(d2_nonlin(y).* (d_W(x,par)*dir1)) * d_xx_W(x,par,dir2) + ...
    diag(d2_nonlin(y).* (d_W(x,par)*dir2)) * d_xx_W(x,par,dir1) + ...
    diag(d2_nonlin(y).* (d_xx_W(x,par,dir2)*dir1)) * d_W(x,par) + ...
    diag( d_nonlin(y)) * d_xxx_W(x,par,dir1,dir2);

end


function [h, W_minus_gamma, b] = test_input(h, W_minus_gamma, b)
if size(W_minus_gamma,1)~=size(W_minus_gamma,2)
    error('The matrix of weights must be square')
end
if size(W_minus_gamma,1)~=size(h,1)
    error('Incompatible input sizes')
end
if nargin < 3
    b = 0;
end
end





