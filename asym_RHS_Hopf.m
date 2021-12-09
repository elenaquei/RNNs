% find Hopf bifurcation in AsymmetricRNN
% Now, we can consider a less symmetric system!
% definition of system-specific elements:

% OLD
dim = 3;
W = @(a) [a, -1, 1;
    1, a, -1;
    -1, 1, a]+ 0.0001 * [ 1,2,3;4,5,6;1,4,8];

% NEW EXAMPLE
dim = 400;
perturbation = 0.1;
if mod(dim,2)==1 && perturbation == 0
    warning('This endeavor would be unsuccessful - skew symmetric matrices of odd dimensions are singular and a non-zero perturbation is needed')
    perturbation = 10^-3;
    fprintf('The perturbation is set to %f\n\n', perturbation);
end
rng('default')
% rng(10)   % CAN FIND PERIODIC ORBIT (in backward time)
% rng(120) % for dim = 50, gives a singular W at alpha = 0.1888, can't
% validate the Hopf associatedd to the largest alpha
rng(80)
validation = 1;
list_finding_orbit_dim6 = [10, 3, 80];
list_finding_orbit_dim50 = [10, 80];
list_finding_orbit_dim150 = [10, 80];
list_finding_orbit_dim400 = [10, 80];
R1 = randn(dim,dim);
R2 = perturbation * randn(dim,dim);
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

% full Hopf problem
F = @(X) wrapper_Hopf(X, dim, f, df, phi);
DF = @(X) wrapper_DHopf(X, dim, df, dxxfv, dalphaf, dxalphaf, phi);

X_merge = @(alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag) [alpha; eigenval_imag; x; eigenvec_real; eigenvec_imag];

% Numerical search for a Hopf bifurcation
solutions = [];
poor_solutions = [];


% getting a better equilibrium for the ODE
x = zeros(size(W(1), 1),1); % system status
alpha = 0;

x = Newton(@(x)f(x,alpha), @(x)df(x,alpha), x);

% at the found equilibrium, find the best eigenpair
[V,L] = eigs(df(x, alpha), dim);

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
            if size(solutions,1)<1 || max(abs(solutions(:,1)-X(1)))>10^-7
                solutions(end+1,:) = X;
                fprintf('A Hopf bifurcation has been found\n')
            end
        else
            fprintf('Newton converged but the norm is too large to be trusted, norm = %e\n', norm(X))
            poor_solutions(end+1,:) = X;
        end
    catch
        error('No Hopf bifurcation found')
    end
    
end

number_of_positive_stab = 0;
number_of_negative_stab = 0;
not_proven = 0;

positive_lyap_index = ([]);
negative_lyap_index = ([]);
positive_lyap = intval([]);
negative_lyap = intval([]);
unproven = [];

fprintf('Finished the numerical search, now the validation starts\n', size(solutions,1))

ddfv = @(x, a, v) dir_der2RHS(x, W(a), v);
dddfvw = @(x, a, v, w) dir_der3RHS(x, W(a), v, w);
bool_val = 1;
dalphaf = @(x,a) der_alpha_RHS(x, W(a), dalphaW(a));
dxalphaf = @(x, a) der_alpha_xRHS(x, W(a), dalphaW(a));
dalphaxxf = @(x, a, v) der_x_x_alpha_RHS(x, W(a), dalphaW(a), v);
dalphaalphaf = @(x, a)der_alpha_alpha_RHS(x, W(a), dalphaW(a), dalphalphaW(a));
dalphaalphaxf = @(x, a, v) der_x_alpha_alpha_RHS(x, W(a), dalphaW(a), dalphalphaW(a));

for i = 1: size(solutions,1)*validation
    solution = solutions(i,:);
    X = solution.';
    try
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
end
if ~isempty(unproven)
    fprintf('We could not prove %i bifurcations\n',length(unproven));
else
    fprintf('We could validate all Hopf bifurcations\n');
end

x = solutions(1:end,2+(1:dim));
bifurcation_values = solutions(1:end,1);
[alpha_big, index] = max(bifurcation_values);
eigenvec  = solutions(1:end,2+dim+(1:dim))+ 1i*solutions(1:end,2+2*dim+(1:dim));
eigenvec_plot = eigenvec(index,:);
plotting_dim = min(dim, 6);
figure
[t,y] = ode45(@(t,x) f(x, alpha_big), [0,500], x(index,:) + sqrt(01000) * abs(eigenvec(2,:)));
plot(t, y(:,1:plotting_dim), 'LineWidth',3)
set(gca,'FontSize',18)
figure
[t,y] = ode45(@(t,x) -f(x, alpha_big), [0,150], x(index,:) + sqrt(0.10) * abs(eigenvec(2,:)));
[t,y] = ode45(@(t,x) f(x, alpha_big), [0,150], y(end,:));
plot(t, y(:,1:plotting_dim),'LineWidth',3)
set(gca,'FontSize',18)
%plot_bifurcation_diag(f, solutions(1:2:end,2+(1:dim)),solutions(1:2:end,1), ...
%    solutions(1:2:end,2+dim+(1:dim))+ 1i*solutions(1:2:end,2+2*dim+(1:dim)))

% case 1
if dim == 6
    figure
    [t,y] = ode45(@(t,x) -f(x, alpha_big), [0,150], x(index,:) + sqrt(0.10) * abs(eigenvec(2,:)));
    [t,y] = ode45(@(t,x) f(x, alpha_big), [0,100], y(end,:));
    plot(t(300:end)-t(300), y(300:end,1:plotting_dim),'LineWidth',3)
    set(gca,'FontSize',18)
elseif dim == 50
    figure
    [t,y] = ode45(@(t,x) -f(x, alpha_big), [0,150], x(index,:) + sqrt(0.10) * abs(eigenvec(2,:)));
    [t,y] = ode45(@(t,x) f(x, alpha_big), [0,45], y(end,:));
    plot(t(1700:end)-t(1700), y(1700:end,1:plotting_dim),'LineWidth',3)
    set(gca,'FontSize',18)
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