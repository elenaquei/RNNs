function [R1, R2,solutions, positive_lyap_index, negative_lyap_index, positive_lyap, negative_lyap, unproven] = asym_sin_RHS_Hopf(dim, varargin)
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
        defaultPerturbation = 0.01;
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
%f = @(x, a) asym_rhs(x, W(a));
%dxf = @(x, a) der_RHS(x, W(a));
%dalphaf = @(x,a) der_alpha_RHS(x, W(a), dalphaW(a));
%dalphaxf = @(x, a) der_alpha_xRHS(x, W(a), dalphaW(a));
%dxxf = @(x, a, v) dir_der2RHS(x, W(a), v);

%  SIN activation function
% [f, dalphaf, dxf, dalphaalphaf, dalphaxf, dxxf, dalphaalphaxf, dalphaxxf, dxxxf] ....
%     = allderivatives(W, dalphaW, dalphalphaW, @(x) sin(x), @(x) cos(x), @(x) -sin(x), @(x) -cos(x));
% TANH activation function
[f, dalphaf, dxf, dalphaalphaf, dalphaxf, dxxf, dalphaalphaxf, dalphaxxf, dxxxf] ....
  = allderivatives(W, dalphaW, dalphalphaW, @(x) tanh(x), @(x) 1-tanh(x).^2, @(x) -2*tanh(x)+2*tanh(x).^3, @(x) -2+8*tanh(x).^2 - 6*tanh(x).^4);

% full Hopf problem - wrappers 
F = @(X) wrapper_Hopf(X, dim, f, dxf, phi);
DF = @(X) wrapper_DHopf(X, dim, dxf, dxxf, dalphaf, dalphaxf, phi);

X_merge = @(alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag) [alpha; eigenval_imag; x; eigenvec_real; eigenvec_imag];

% Numerical search for a Hopf bifurcation
solutions = [];
poor_solutions = [];


% equilibrium for the ODE
alpha = 0;
x = zeros(size(W(1), 1),1); 
x = Newton(@(x)f(x,alpha), @(x)dxf(x,alpha), x);

% at the found equilibrium, find the best eigenpair
[V,L] = eigs(dxf(x, alpha), dim);

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
%dxf = @(x, a, v) dir_der2RHS(x, W(a), v);
%dxxxf = @(x, a, v, w) dir_der3RHS(x, W(a), v, w);
bool_val = 1;
%dalphaf = @(x,a) der_alpha_RHS(x, W(a), dalphaW(a));
%dalphaxf = @(x, a) der_alpha_xRHS(x, W(a), dalphaW(a));
%dalphaxxf = @(x, a, v) der_x_x_alpha_RHS(x, W(a), dalphaW(a), v);
%dalphaalphaf = @(x, a)der_alpha_alpha_RHS(x, W(a), dalphaW(a), dalphalphaW(a));
%dalphaalphaxf = @(x, a, v) der_x_alpha_alpha_RHS(x, W(a), dalphaW(a), dalphalphaW(a));


% validation loop
for i = 1: size(solutions,1)*(validation~=0)
    solution = solutions(i,:);
    X = solution.';
    try
        % Validation - including computation of first Lyapunov coeff
        [x_star,lambda_star,eigenvec,eigenval, l1] = ...
            algebraic_hopf(f,dxf,dxxf,dxxxf, dalphaf, dalphaxf,dalphaxxf, dalphaalphaf,dalphaalphaxf, dim,X,phi,bool_val);
        if l1>0
            number_of_positive_stab = number_of_positive_stab+1;
            positive_lyap_index(end+1) = i;
            positive_lyap(end+1) = l1;
        else
            number_of_negative_stab = number_of_negative_stab+1;
            negative_lyap_index(end+1)= i;
            negative_lyap(end+1)= l1;
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
    fprintf('We could validate all Hopf bifurcations, with %i positive and %i negative\n', number_of_positive_stab,number_of_negative_stab);
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


function [f_x, dalphaf, dxf, dalphaalphaf, dalphaxf, dxxf, dalphaalphaxf, dalphaxxf, dxxxf, error_d3] ....
    = allderivatives(W, W_prime, W_second, f, f_prime, f_second, f_third)

f_x =@(x,a) f(W(a) * x);

dalphaf =@(x,a) diag(f_prime(W(a)*x))*(W_prime(a)*x);
dxf =@(x,a) diag(f_prime(W(a)*x)) * W(a);

dalphaalphaf =@(x,a) f_second(W(a)*x).*(W_prime(a)*x).*(W_prime(a)*x) + diag(f_prime(W(a)*x)) * W_second(a)*x;
dalphaxf =@(x,a) diag(f_second(W(a)*x).*(W_prime(a)* x)) * W(a) + diag(f_prime(W(a)*x)) * W_prime(a); % not checked
dxxf =@(x,a, v) diag(f_second(W(a)*x) .* (W(a)*v)) * W(a);

dalphaalphaxf =@(x,a) diag(f_third(W(a)*x) .* (W_prime(a)*x) .* (W_prime(a)*x)) * W(a) ...
    + diag(f_second(W(a)*x) .* (W_second(a)*x)) * W(a) ...
    + 2 * diag(f_second(W(a)*x) .* (W_prime(a)*x)) * W_prime(a)...
    + diag(f_prime(W(a)*x))*W_second(a);

dalphaxxf =@(x,a,v) diag(f_third(W_prime(a)*x).*(W(a)*v).*(W(a)*x)) * W(a) ...
    + diag(f_second(W(a)*x).* (W_prime(a)*v)) * W(a)...
    + diag(f_second(W(a)*x) .* (W(a)*v)) * W_prime(a);

dxxxf = @(x,a,v,w)diag(f_third(W(a)*x).* (W(a)*v).*(W(a)*w))* W(a);

%     function dxxxf = der3f(W,x,a,v,w)
%         dxxxf = diag(f_third(W(a)*x).* (W(a)*v).*(W(a)*w))* W(a);
%         dxxxf = diag(-2+8*tanh(W(a)*x).^2 - 6*tanh(W(a)*x).^4) * diag(W(a) * v) * diag(W(a) * w)*W(a);
%     end
% 
% dxxxf = @(x,a,v,w)der3f(W,x,a,v,w);

% this is reliably 0
% error_d3 = @(x,a,v,w) der3f(W,x,a,v,w) - diag(f_third(W(a)*x).* (W(a)*v).*(W(a)*w))* W(a);

end

