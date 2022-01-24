function [R1, R2,solutions, positive_lyap_index, negative_lyap_index, positive_lyap, negative_lyap, unproven] = asym_RHS_Hopf(dim, varargin)
% find Hopf bifurcation in AsymmetricRNN
% This code generates a random dynamical system based on RNNs, then
% numerically finds and validates Hopf bifurcations w.r.t. the
% hyperparameter.
% This code refers to the paper "PLACEHOLDER" by E. Queirolo and C. Kuehn
%
% function asym_RHS_Hopf(dim, varargin)
% can be called with additional info
% perturbation (DEFAULT 0.1), R1 and R2, or seed (DEFAULT 80)
% 
% Examples
% asym_RHS_Hop(10)
% asym_RHS_Hopf(10, 'seed', 4)
% asym_RHS_Hopf(4, 'perturbation', 0.01)
% asym_RHS_Hopf(6, 'perturbation', 10^-4)
% asym_RHS_Hopf(100, 'R1', R1, 'R2', R2) where R1 and R2 are square
%                                           matrices of same size
% combination of inputs is also possible, such as
%
% asym_RHS_Hopf(10, 'perturbation', 0.4, 'R1', R1, 'seed', 7)
%
% OUTPUTS
%
% solutions     each column is a numerical solution, stored as 
%               alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag
% positive_lyap_index, negative_lyap_index, positive_lyap, negative_lyap, unproven

    function [dim, seed, R1, R2, validation] = input_parse(dim, varargin)
        defaultPerturbation = 0.1;
        defaultSeed = 80;
        defaultValidation = 1;
        defaultR1 = zeros(dim, dim);
        defaultR2 = zeros(dim, dim);

        p = inputParser;
        validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
        validInteger = @(x) isnumeric(x) && isscalar(x) && (mod(x,1) ==0);
        addOptional(p,'perturbation',defaultPerturbation,validScalarPosNum);
        addOptional(p,'validation',defaultValidation,validInteger);
        addOptional(p,'seed',defaultSeed,validInteger);
        validMatrix = @(x) isnumeric(x) && size(x,1) == dim && size(x,2) == dim && length(size(x))==2;
        addParameter(p,'R1',defaultR1,validMatrix);
        addParameter(p,'R2',defaultR2,validMatrix);
        parse(p,dim,varargin{:});
        perturbation = p.Results.perturbation;
        validation = p.Results.validation;
        seed = p.Results.seed;
        R1 = p.Results.R1;
        R2 = p.Results.R2;
        
        rng('default')
        rng(seed)
        
        if max(abs(R1))==0
        	R1 = randn(dim, dim);
        end
        if max(abs(R2))==0
        	R2 = perturbation * randn(dim, dim);
        end
        %parse(p,dim,varargin{:});
        

        if mod(dim,2)==1 && perturbation == 0
            warning('This endeavor would be unsuccessful - skew symmetric matrices of odd dimensions are singular and a non-zero perturbation is needed')
            perturbation = 10^-3;
            fprintf('The perturbation is set to %f\n\n', perturbation);
            R1 = randn(dim,dim);
            R2 = perturbation * randn(dim,dim);
        end
    end

% inout parsing and definition of ystem parameters
[dim, seed, R1, R2, validation] = input_parse(dim, varargin{:});

% rng(120) % for dim = 50, gives a singular W at alpha = 0.1888, can't
% validate the Hopf associatedd to the largest alpha

% tested random seeds 
list_finding_orbit_dim6 = [10, 3, 80];
list_finding_orbit_dim50 = [10, 80];
list_finding_orbit_dim150 = [10, 80];
list_finding_orbit_dim400 = [10, 80];

% definition of the full matrix 
W = @(a) R1 - R1.' + a * eye(dim) + R2;
dalphaW = @(a) eye(dim);
dalphalphaW = @(a) zeros(dim,dim);
phi = randi([-5,5],[dim,1]);
phi = phi/norm(phi);

% construction of the Hopf problem based on the just defined elements:
f = @(x, a) asym_rhs(x, W(a));
df = @(x, a) der_RHS(x, W(a));
dalphaf = @(x,a) der_alpha_RHS(x, W(a), dalphaW(a));
dxalphaf = @(x, a) der_alpha_xRHS(x, W(a), dalphaW(a));
dxxfv = @(x, a, v) dir_der2RHS(x, W(a), v);

% full Hopf problem - wrappers 
F = @(X) wrapper_Hopf(X, dim, f, df, phi);
DF = @(X) wrapper_DHopf(X, dim, df, dxxfv, dalphaf, dxalphaf, phi);

X_merge = @(alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag) [alpha; eigenval_imag; x; eigenvec_real; eigenvec_imag];

% Numerical search for a Hopf bifurcation
solutions = [];
poor_solutions = [];


% equilibrium for the ODE
alpha = 0;
x = zeros(size(W(1), 1),1); 
x = Newton(@(x)f(x,alpha), @(x)df(x,alpha), x);

% at the found equilibrium, find the best eigenpair
[V,L] = eigs(df(x, alpha), dim);

% loop through eigenpairs to find associated Hopf bifurcations
for i = 1:dim
    eigenval_imag = imag(L(i,i));
    
    % rotate the eigenvector to fit the linear scaling
    first_eig = V(:,i);
    scaling = 1i / (phi.' * first_eig);
    first_eig = scaling * first_eig;
    eigenvec_real = real(first_eig);
    eigenvec_imag = imag(first_eig);
    
    X = X_merge(alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag);
    
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
ddfv = @(x, a, v) dir_der2RHS(x, W(a), v);
dddfvw = @(x, a, v, w) dir_der3RHS(x, W(a), v, w);
bool_val = 1;
dalphaf = @(x,a) der_alpha_RHS(x, W(a), dalphaW(a));
dxalphaf = @(x, a) der_alpha_xRHS(x, W(a), dalphaW(a));
dalphaxxf = @(x, a, v) der_x_x_alpha_RHS(x, W(a), dalphaW(a), v);
dalphaalphaf = @(x, a)der_alpha_alpha_RHS(x, W(a), dalphaW(a), dalphalphaW(a));
dalphaalphaxf = @(x, a, v) der_x_alpha_alpha_RHS(x, W(a), dalphaW(a), dalphalphaW(a));

% validation loop
for i = 1: size(solutions,1)*(validation~=0)
    solution = solutions(i,:);
    X = solution.';
    try
        % Validation - including computation of first Lyapunov coeff
        [x_star,lambda_star,eigenvec,eigenval, l1] = ...
            algebraic_hopf(f,df,ddfv,dddfvw, dalphaf, dxalphaf,dalphaxxf, dalphaalphaf,dalphaalphaxf, dim,X,phi,bool_val);
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



function plot_bifurcation_diag(f, x, alpha, eigenvec)
% unused because unreliable
figure
iters = 50;
amplitudes = zeros(length(alpha), iters);
stepsize = 10^-6;
for i = 1:length(alpha)
    previous_orbit = x(i,:);
    for j = 1:iters
        dist_bif = j*stepsize;
        alpha_iter = alpha(i) + dist_bif;
        if j < 10
            epsilon = sqrt(dist_bif);
            [amplitude, end_orbit] = detect_amplitude(@(x)f(x,alpha_iter), x(i,:), x(i,:) + epsilon * real(eigenvec), 200, 0);
        else
            [amplitude, end_orbit] = detect_amplitude(@(x)f(x,alpha_iter), x(i,:), previous_orbit, 100, 10^-6);
        end
        amplitudes(i,j) = amplitude;
        previous_orbit = end_orbit;   
    end
    plot(alpha(i) + (1:iters)*stepsize, amplitudes(i,:), 'b')
    hold on
end
plot(alpha,0*alpha, 'r*')
end
