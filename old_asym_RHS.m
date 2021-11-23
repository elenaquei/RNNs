% find Hopf bifurcation in AsymmetricRNN
if 2==1
    for j = -1:0.1:1
        W = [ 0, j; - j, 0];
        h = [0;0];
        gamma = 0.15;
        eig = eigs(der_RHS(h,W, gamma));
        plot(j,imag(eig(1)),'*');
        hold on
        plot(j, real(eig(1)), '.')
    end
end

% finding Hopf bifurcations by hand is possible, but the simplest system
% has a degenerate Hopf bifurcation : dxf = 0, d_alpha_f = 0
% we need to go towards abigger system

% test system to check all derivatives:

gamma = 0.15;
W = @(a) [0, a;
    -a, 0] + gamma * eye(2);
dalphaW = @(a) [0, 1;-1,0];

f = @(x, a) asym_rhs(x, W(a));
df = @(x, a) der_RHS(x, W(a));
dalphaf = @(x,a) der_alpha_RHS(x, W(a), dalphaW(a));
dxalphaf = @(x, a) der_alpha_xRHS(x, W(a), dalphaW(a));
dxxfv = @(x, a, v) dir_der2RHS(x, W(a), v);
x =  rand(size(W(1), 1),1); % system status
alpha = 0.3; % parameter
test = f(x, alpha);
testdf = df(x, alpha);

phi = [1;2];
dim = 2;
F = @(X) wrapper_Hopf(X, dim, f, df, phi);
DF = @(X) wrapper_DHopf(X, dim, df, dxxfv, dalphaf, dxalphaf, phi);

X_merge = @(alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag) [alpha; eigenval_imag; x; eigenvec_real; eigenvec_imag];

[V,L] = eigs(df(x, alpha));
eigenval_imag = imag(L(1,1));
first_eig = V(:,1);
scaling = 1i / (phi.' * first_eig);
first_eig = scaling * first_eig;
eigenvec_real = real(first_eig);
eigenvec_imag = imag(first_eig);

X = X_merge(alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag);

% test of appropriate computation of derivatives:
F(X);
analytic_DF = DF(X);
numerical_Df = finite_diff_fn(F,X);

if norm(analytic_DF- numerical_Df)>10^-5
    error('Some derivatives are wrong')
end

% don't do Newton: the bifurcation is singular!!
% [X]=Newton(F, DF, X);

% Now, we can consider a bigger system!
% definition of system-specific elements:
solutions = [];
poor_solutions = [];
W = @(a) [a, -1, 1;
    1, a, -1;
    -1, 1, a]+ 0.0001 * [ 1,2,3;4,5,6;1,4,8];
dalphaW = @(a) [1, 0, 0;
    0,1,0;
    0,0,1];
dalphalphaW = @(a) zeros(3,3);
phi = [1;2;-1];
dim = 3;

% construction of the Hopf problem based on the just defined elements:
f = @(x, a) asym_rhs(x, W(a));
df = @(x, a) der_RHS(x, W(a));
dalphaf = @(x,a) der_alpha_RHS(x, W(a), dalphaW(a));
dxalphaf = @(x, a) der_alpha_xRHS(x, W(a), dalphaW(a));
dxxfv = @(x, a, v) dir_der2RHS(x, W(a), v);

% full Hopf problem
F = @(X) wrapper_Hopf(X, dim, f, df, phi);
DF = @(X) wrapper_DHopf(X, dim, df, dxxfv, dalphaf, dxalphaf, phi);

X_merge = @(alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag) [alpha; eigenval_imag; x; eigenvec_real; eigenvec_imag];


for alpha = 0.5
    if mod(alpha, 20) ==0
        fprintf('testing alpha = %f\n', alpha);
    end
    
    % getting a better equilibrium for the ODE
    x = zeros(size(W(1), 1),1); % system status
    for i = 1:10
        try
            x = Newton(@(x)f(x,alpha), @(x)df(x,alpha), x);
            break
        catch
            x = 10*rand(size(W(1), 1),1)-5;
            continue
        end
    end
    if i < 10
        %fprintf('alpha = %f gives a good equilibrium  [%f,%f,%f]\n', alpha, x(1), x(2), x(3))
    else 
        fprintf('      alpha = %f fails to find an equilibrium', alpha)
        continue
    end
    
    % at the found equilibrium, find the best eigenpair
    [V,L] = eigs(df(x, alpha));
    eigenval_imag = imag(L(1,1));
    
    % rotate the eigenvector to fit the linear scaling
    first_eig = V(:,1);
    scaling = 1i / (phi.' * first_eig);
    first_eig = scaling * first_eig;
    eigenvec_real = real(first_eig);
    eigenvec_imag = imag(first_eig);
    
    X = X_merge(alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag);
    
    % Newton to find Hopf bifurcation
    try
        [X]=Newton(F, DF, X);
        if log(norm(X))<10
            solutions(end+1,:) = X;
            fprintf('A Hopf bifurcation has been found\n')
        else
            fprintf('Newton converged but the norm is too large to be trusted, norm = %e\n', norm(X))
            poor_solutions(end+1,:) = X;
        end
    catch
        continue
    end
end

disp('End')
if size(solutions,1)>0
    solution = solutions(1,:);
    X = solution.';
    ddfv = @(x, a, v) dir_der2RHS(x, W(a), v);
    dddfvw = @(x, a, v, w) dir_der3RHS(x, W(a), v, w);
    bool_val = 0;
    dalphaf = @(x,a) der_alpha_RHS(x, W(a), dalphaW(a));
    dxalphaf = @(x, a) der_alpha_xRHS(x, W(a), dalphaW(a));
    dalphaxxf = @(x, a, v) der_x_x_alpha_RHS(x, W(a), dalphaW(a), v);
    dalphaalphaf = @(x, a)der_alpha_alpha_RHS(x, W(a), dalphaW(a), dalphalphaW(a));
    dalphaalphaxf = @(x, a, v) der_x_alpha_alpha_RHS(x, W(a), dalphaW(a), dalphalphaW(a));
    
    [x_star,lambda_star,eigenvec,eigenval, stability] = ...
      algebraic_hopf(f,df,ddfv,dddfvw, dalphaf, dxalphaf,dalphaxxf, dalphaalphaf,dalphaalphaxf, dim,X,phi,bool_val);
end

function [alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag] = split_into_elements(X, dim)
if length(X) ~= 3*dim+2
    error('Sizes are not compatible')
end

alpha = X(1);
eigenval_imag = X(2);
x = X(2+(1:dim));
eigenvec_real = X(dim+3:2*dim+2);
eigenvec_imag = X(2*dim+3:3*dim+2);
end


function y = wrapper_Hopf(X, dim, f, df, phi)
[alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag] = split_into_elements(X, dim);
y = f_Hopf(f, df, phi, alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag);
end

function DF = wrapper_DHopf(X, dim, df, ddf, dalphaf, dxalphaf, phi)
[alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag] = split_into_elements(X, dim);
DF = Df_Hopf([], df, ddf, dalphaf, dxalphaf, phi, alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag);
end


function h_dot = asym_rhs(varargin)
% INPUT: h, W, gamma, b
[h, W_minus_gamma, b] = test_input(varargin{:});

I = eye(size(W_minus_gamma,1));

h_dot = tanh( W_minus_gamma * h + b);
end

function der = der_RHS(varargin)
[h, W_minus_gamma, b] = test_input(varargin{:});

y = W_minus_gamma*h + b;
der = diag(1- tanh(y).^2) * W_minus_gamma;
end

function dalphaRHS =  der_alpha_RHS(x, W_minus_gamma, dalphaW)

y = W_minus_gamma*x;

tanh_prime = @(y) 1 - tanh(y).^2;

dalphaRHS = diag(tanh_prime(y)) * dalphaW * x;
end


function dalphaxRHS = der_alpha_xRHS(x, W_minus_gamma, dalphaW)

v = dalphaW * x;

y = W_minus_gamma*x;

tanh_prime = @(y) 1 - tanh(y).^2;

dalphaxRHS = diag(  - 2 * tanh(y) .* tanh_prime(y) .* v) * W_minus_gamma + diag( tanh_prime(y) ) * dalphaW;

end



function dxxfv = dir_der2RHS(x, W_minus_gamma, v)
y = W_minus_gamma* x;

dxxfv = - diag(2 * tanh(y) .*(1-tanh(y).^2) .* (W_minus_gamma * v)) * W_minus_gamma;

end


function dxxalphafv = der_x_x_alpha_RHS(x, W_minus_gamma, W_prime, v)
y = W_minus_gamma* x;
y_prime = W_prime*x;
tanh_prime = 1 - tanh(y).^2;

diagonal = tanh_prime .* y_prime .* (W_minus_gamma*v) - 3 * tanh(y).^2 .* tanh_prime.* y_prime .* (W_minus_gamma*v) + tanh(y).*tanh_prime.* (W_prime * v);

dxxalphafv = - 2*diag(diagonal) * W_minus_gamma - 2 * diag( tanh(y).*tanh_prime.*(W_minus_gamma*v))* W_prime;

end


function dxalphaalphafv = der_x_alpha_alpha_RHS(x, W_minus_gamma, W_prime, W_second)
y = W_minus_gamma * x;
y_prime = W_prime * x;
y_second = W_second * x;
tanh_prime = 1 - tanh(y).^2;
tanh_second = 2*tanh(y).*tanh_prime;

diagonal1 = tanh_prime.^2 .* y_prime.^2 + tanh(y).*tanh_second.*y_prime.^2 + tanh(y) .*tanh_prime.*y_second;
diagonal2 = -2*tanh(y).*tanh_prime.*y_prime;


dxalphaalphafv = - 2*diag(diagonal1) * W_minus_gamma + diag(diagonal2)* W_prime + diag(tanh_prime)*W_second;

end

function dalphaalphafv = der_alpha_alpha_RHS(x, W_minus_gamma, W_prime, W_second)
y = W_minus_gamma * x;
y_prime = W_prime * x;
y_second = W_second * x;
tanh_prime = 1 - tanh(y).^2;
tanh_second = 2*tanh(y).*tanh_prime;

dalphaalphafv = tanh_second.*y_prime.^2 + tanh_prime.*y_second;

end


function dxxxfvw = dir_der3RHS(x, W_minus_gamma, v, w)
y = W_minus_gamma* x;
tanh_prime = 1 - tanh(y).^2;
tanh_doupbleprime = -2 * tanh(y).*tanh_prime;

dxxxfvw = -2 * diag(tanh_prime + tanh(y).*tanh_doupbleprime) * diag(W_minus_gamma * v) * diag(W_minus_gamma * w);
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