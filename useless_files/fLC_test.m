% numerical computation of the first Lyapunov exponent in a given ODE
% system (finite time) 
% could there be a cool way of going to infinity? No idea

debug = 1;
T_final = 30;
close all
% definition of right hand side

% Example 1: the Lorenz butterfly. The fLC should tend to 0.9056
sigma = 10; beta = 8/3; rho = 28;
f = @(t,x) [sigma * (x(2,:) - x(1,:));
x(1,:).*(rho - x(3,:)) - x(2,:);
x(1,:) .* x(2,:) - beta * x(3,:)];

Jac = @(t,x)[ - sigma, sigma, 0
    rho-x(3,:), -1, -x(1,:)
    x(2,:), x(1,:), -beta];

y0 = [-1.2595115397689e+01  -1.6970525307084e+01   27]';
approx_period = 2.3059072639398e+00;
T_final = approx_period;
if debug>3
    [t,y] = ode45(f, [0,T_final],y0);
    plot3(y(:,1), y(:,2), y(:,3))
end

W = @(Q, J) (Q' * J * Q);

big_system = @(t, X) [f(t,X(1:3,:))
    Jac(t, X(1:3,:)) * X(4:6,:) - X(4:6) * W(X(4:6), Jac(t,X(1:3)));
    W(X(4:6), Jac(t,X(1:3)))];

e1 = 0*y0; e1(1) = 1;

initial_cond = [y0; e1; 1];
[t,y] = ode45(big_system, [0,T_final],initial_cond);
if debug
    figure
    plot3(y(:,1), y(:,2), y(:,3))
end

figure
plot(y(1000:end,end)./t(1000:end))
finite_time_fLc = y(end,end)./T_final;
fprintf('The approximate finite time first Lyapunov exponent is %f\n', finite_time_fLc)
fprintf('The  following validation has not been successful\n'/)

% start VALIDATION

% first, validate a simple lorenz orbit
global nu
global use_intlab
global talkative
global RAD_MAX
global Display
global norm_weight
norm_weight = [];
Display = 1;
talkative = 10;
use_intlab = 0;
RAD_MAX = 10^-2;

try
    intval(1);
catch
    addpath(genpath('../'))
    startintlab
end

% problem dependent
nu = 1.05;
pho_null = 28;
n_nodes = 50;
step_size = 10^-4;
sigma = 10;
beta = 8/3;
pho = 28;
s = 'lorenz_validation'; % path where the validation will be saved

% construct the numerical solution with forward integration from known
% initial conditions
init_coord  = [-1.2595115397689e+01  -1.6970525307084e+01   27];
approx_period = 2.3059072639398e+00;

% right hand side
f=@(t,x)[sigma*(x(2) - x(1)); x(1)*(pho-x(3)) - x(2); x(1)*x(2)- beta*x(3)];

% forward integration
[tout, yout] = ode45(f,[0,approx_period],init_coord);

% transformation to Xi_vector from time series
xXi = time_series2Xi_vec(tout,yout,n_nodes);


% definition of the vector field in the form of a string
string_lorenz = '- dot x1 + sigma l1 x2 - sigma l1 x1 \n - dot x2 + pho l1 x1 - l1 x1 x3 - l1 x2 \n - dot x3 + l1 x1 x2 - beta l1 x3\n'; % general lorenz
lyapunov_problem_string = strrep(string_lorenz, 'sigma' , num2str(sigma)); % plugging in sigma
lyapunov_problem_string = strrep(lyapunov_problem_string, 'beta' , num2str(beta)); % plugging in beta
% fixed point vector field
string_lorenz_pho = strrep(lyapunov_problem_string, 'pho', num2str(pho_null)); % for point wise first system, plugging in pho
% continuation vector field, where pho is the second scalar variable
string_lorenz_cont = strrep(lyapunov_problem_string, 'pho', 'l2'); % setting pho as the second scalar variable

% constructing the ODEs systems
% fixed pho
scal_eq = default_scalar_eq(xXi);
polynomial_fix = from_string_to_polynomial_coef(string_lorenz_pho);
F_fix = full_problem(scal_eq, polynomial_fix);

% refining the approximation with Newton method
sol = Newton_2(xXi,F_fix,30,10^-7);

% defining the problem
scal_eq = default_scalar_eq(sol);
polynomial = from_string_to_polynomial_coef(string_lorenz_pho); 
F_square = full_problem(scal_eq, polynomial);

% launch the validation
success = validation_orbit(F_square, sol);

if ~success
    error('The validation failed and there must be a mistake somewhere')
end

% set up the full problem
W_string = '-sigma q1^2 + pho q1q2 - x3 q1q2 + x2 q1q3 + sigma q1q2 - q2^2 - beta q3^2';
Q1_dot = '-dot q1 - sigma q1 + sigma q2 -sigma q1^3 + pho q1^2q2 - x3 q1^2q2 + x2 q1^2q3 + sigma q1^2q2 - q1q2^2 - beta q1q3^2';
Q2_dot = '-dot q2 + pho q1 - x3 q1 - q2 - x1 q3 -sigma q1^2q2 + pho q1q2^2 - x3 q1q2^2 + x2 q1q2q3 + sigma q1q2^2 - q2^3 - beta q2q3^2';
Q3_dot = '-dot q3 + x2q1 + x1q2 - beta q3 -sigma q1^2q3 + pho q1q2q3 - x3 q1q2q3 + x2 q1q3^2 + sigma q1q2q3 - q2^2q3 - beta q3^3';
Q_dot = append(Q1_dot, '\n', Q2_dot, '\n', Q3_dot, '\n');
rho_dot = append('-dot rho1 ',W_string);
lyapunov_problem_string = append(string_lorenz, Q_dot, rho_dot);

lyapunov_problem_string = strrep(lyapunov_problem_string, 'sigma' , num2str(sigma)); % plugging in sigma
lyapunov_problem_string = strrep(lyapunov_problem_string, 'beta' , num2str(beta)); % plugging in beta
% fixed point vector field
lyapunov_problem_string = strrep(lyapunov_problem_string, 'pho', num2str(pho_null));

lyapunov_problem_string = strrep(lyapunov_problem_string, 'q1', 'x4');
lyapunov_problem_string = strrep(lyapunov_problem_string, 'q2', 'x5');
lyapunov_problem_string = strrep(lyapunov_problem_string, 'q3', 'x6');
lyapunov_problem_string = strrep(lyapunov_problem_string, 'rho1', 'x7');

% compute the full solution of the larger problem
xXi = time_series2Xi_vec(t,y,n_nodes);

scal_eq = default_scalar_eq(xXi);
polynomial_fix = from_string_to_polynomial_coef(lyapunov_problem_string);
F_fix = full_problem(scal_eq, polynomial_fix);


% test that the computed Lyapunov exponent makes sense 
