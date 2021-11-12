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

W = @(a) [0, a;
    -a, 0];
dalphaW = @(a) [0, 1;-1,0];
gamma = 0.15;

f = @(x, a) asym_rhs(x, W(a), gamma);
df = @(x, a) der_RHS(x, W(a), gamma);
dalphaf = @(x,a) der_alpha_RHS(x, W(a), gamma, dalphaW(a));
dxalphaf = @(x, a) der_alpha_xRHS(x, W(a), gamma, dalphaW(a));
dxxfv = @(x, a, v) dir_der2RHS(x, W(a), gamma, v);
x =  rand(size(W(1), 1),1); % system status
alpha = 0.3; % parameter
test = f(x, alpha);
testdf = df(x, alpha);

phi = [1;2];
dim = 2;
F = @(X) wrapper_Hopf(X, dim, f, df, phi);
DF = @(X) wrapper_DHopf(X, dim, df, dxxfv, dalphaf, dxalphaf, phi);

X_merge = @(x, alpha, eigenvec_real, eigenvec_imag, eigenval_imag) [x; alpha; eigenvec_real; eigenvec_imag; eigenval_imag];

[V,L] = eigs(df(x, alpha));
eigenval_imag = imag(L(1,1));
first_eig = V(:,1);
scaling = 1i / (phi.' * first_eig);
first_eig = scaling * first_eig;
eigenvec_real = real(first_eig);
eigenvec_imag = imag(first_eig);

X = X_merge(x, alpha, eigenvec_real, eigenvec_imag, eigenval_imag);

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
W = @(a) [0, 0, a;
    0, 0, 0;
    -a, 0, 0];
dalphaW = @(a) [0, 0, 1;
    0,0,0;
    -1,0,0];
gamma = 0.15;
phi = [1;2;3];
dim = 3;

% construction of the Hopf problem based on the just defined elements:
f = @(x, a) asym_rhs(x, W(a), gamma);
df = @(x, a) der_RHS(x, W(a), gamma);
dalphaf = @(x,a) der_alpha_RHS(x, W(a), gamma, dalphaW(a));
dxalphaf = @(x, a) der_alpha_xRHS(x, W(a), gamma, dalphaW(a));
dxxfv = @(x, a, v) dir_der2RHS(x, W(a), gamma, v);

% full Hopf problem
F = @(X) wrapper_Hopf(X, dim, f, df, phi);
DF = @(X) wrapper_DHopf(X, dim, df, dxxfv, dalphaf, dxalphaf, phi);

X_merge = @(x, alpha, eigenvec_real, eigenvec_imag, eigenval_imag) [x; alpha; eigenvec_real; eigenvec_imag; eigenval_imag];


for alpha = -100:0.5:100
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
    
    X = X_merge(x, alpha, eigenvec_real, eigenvec_imag, eigenval_imag);
    
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
% [x_star,lambda_star,eigenvec,eigenval, stability] = ...
%     algebraic_hopf(f,df,ddf,dddf,N,X,phi,bool_val);


function [x, alpha, eigenvec_real, eigenvec_imag, eigenval_imag] = split_into_elements(X, dim)
if length(X) ~= 3*dim+2
    error('Sizes are not compatible')
end
x = X(1:dim);
alpha = X(dim+1);
eigenvec_real = X(dim+2:2*dim+1);
eigenvec_imag = X(2*dim+2:3*dim+1);
eigenval_imag = X(end);
end


function y = wrapper_Hopf(X, dim, f, df, phi)
[x, alpha, eigenvec_real, eigenvec_imag, eigenval_imag] = split_into_elements(X, dim);
y = f_Hopf(f, df, phi, x, alpha, eigenvec_real, eigenvec_imag, eigenval_imag);
end

function DF = wrapper_DHopf(X, dim, df, ddf, dalphaf, dxalphaf, phi)
[x, alpha, eigenvec_real, eigenvec_imag, eigenval_imag] = split_into_elements(X, dim);
DF = Df_Hopf([], df, ddf, dalphaf, dxalphaf, phi, x, alpha, eigenvec_real, eigenvec_imag, eigenval_imag);
end


function h_dot = asym_rhs(varargin)
% INPUT: h, W, gamma, b
[h, W, gamma, b] = test_input(varargin{:});

I = eye(size(W,1));

h_dot = tanh( (W - gamma * I ) * h + b);
end

function der = der_RHS(varargin)
[h, W, gamma, b] = test_input(varargin{:});

I = eye(size(W,1));

y = (W - gamma * I)*h + b;
der = diag(1- tanh(y).^2) * (W - gamma * I);
end

function dalphaRHS =  der_alpha_RHS(x, W, gamma, dalphaW)
I = eye(size(W,1));

y = (W - gamma * I)*x;

tanh_prime = @(y) 1 - tanh(y).^2;

dalphaRHS = diag(tanh_prime(y)) * dalphaW * x;
end


function dalphaxRHS = der_alpha_xRHS(x, W, gamma, dalphaW)

v = dalphaW * x;

I = eye(size(W,1));
y = (W - gamma * I)*x;

tanh_prime = @(y) 1 - tanh(y).^2;

dalphaxRHS = diag(  - 2 * tanh(y) .* tanh_prime(y) .* v) * (W - gamma * I ) + diag( tanh_prime(y) ) * dalphaW;

end

function dxxfv = dir_der2RHS(x, W, gamma, v)
I = eye(size(W,1));
y = (W - gamma * I)* x;

dxxfv = - diag(2 * tanh(y) .*(1-tanh(y).^2) .* (( W - gamma * I ) * v)) * (W-gamma*I);

end

function [h, W, gamma, b] = test_input(h, W, gamma, b)
if size(W,1)~=size(W,2)
    error('The matrix of weights must be square')
end
if size(W,1)~=size(h,1)
    error('Incompatible input sizes')
end
if nargin< 4
    b =0;
end
if nargin < 3
    gamma = 0;
end
if any(diag(W)~=0)
    W = W - W.';
end
end