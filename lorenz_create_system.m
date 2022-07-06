% from Lorenz to FTFLE

string_lorenz = "sigma*(y-x), x*(rho-z) - y, x*y - beta * z";
Dlorenz = '-sigma, sigma, 0, \n rho-z, -1, -x, \n y, x, -beta';

W = "sigma * q1*(q2-q1) + q1*q2*(rho - z) - q2^2+ q1*q3*y-beta*q3^2";

q_dot = strcat("-sigma* q1 + sigma *q2 - q1*(",W,"), (rho - z)*q1 - q2 - x*q3 - q2*(",W,"), q1*y+q2*x-beta*q3-q3*(",W,")");

p_dot = W;

full_system = strcat(string_lorenz, ',', q_dot, ',', p_dot);

variables = 'x, y, z, q1, q2, q3, p';

c(1) = 0.;
c(2) = -8.3809417428298;
c(3) = 0.029590060630665;

% sigma, beta, rho = (10),  (8)/3, (28)
sigma = 10;
beta = 8/3;
rho = 28;

f = @(x) [sigma*(x(2)-x(1)), x(1)*(rho-x(3)) - x(2), x(1)*x(2) - beta * x(3)];
Df = @(x) [-sigma, sigma, 0; rho-x(3), -1, -x(1); x(2), x(1), -beta];

[l, v]= eigs(Df(c));

% coupling: term epsilon
variables_x = 'x_i, y_i, z_i';
string_lorenz = "sigma*(y_i - x_i + epsilon * x_i+1), x_i*(rho-z_i) - y_i, x_i*y_i - beta * z_i";

Dlorenz_diag = '-sigma, sigma, 0, \n rho-z_i, -1, -x_i, \n y_i, x_i, -beta';
Dlorenz_off_diag = 'sigma*epsilon, 0, 0,\n 0,0,0,\n 0,0,0';

variables_q = 'q_i, r_i, s_i';
W_qrs = "sigma * q_i*(r_i-q_i) + q_i*r_i*(rho - z_i) - r_i^2+ q_i*s_i*y_i-beta*s_i^2 + q_i+1 * sigma * epsilon * q_i"; % new unknowns q r and s

dim = 2;
coupled_lorenz = '';
coupled_W = '';
coupled_q_dot = '';
coupled_variables_x = '';
coupled_variables_q = '';

for i = 1:dim
    temp_lor = assign_index(string_lorenz, i, dim);
    temp_W = assign_index(W_qrs, i, dim);
    temp_var_x = assign_index(variables_x, i, dim);
    temp_var_q = assign_index(variables_q, i, dim);
    
    if i < dim
    coupled_lorenz = strcat(coupled_lorenz, temp_lor,',');
    coupled_W = strcat(coupled_W, temp_W,'+');
    coupled_variables_x = strcat(coupled_variables_x, temp_var_x,',');
    coupled_variables_q = strcat(coupled_variables_q, temp_var_q,','); 
    else
    coupled_lorenz = strcat(coupled_lorenz, temp_lor);
    coupled_W = strcat(coupled_W, temp_W);
    coupled_variables_x = strcat(coupled_variables_x, temp_var_x);
    coupled_variables_q = strcat(coupled_variables_q, temp_var_q); 
    end
end

q_dot_qrs = strcat("-sigma* q_i + sigma *r_i - q_i*(",coupled_W,...
    "), (rho - z_i)*q_i - r_i - x_i*s_i - r_i*(",coupled_W,...
    "), q_i*y_i+r_i*x_i-beta*s_i-s_i*(",coupled_W,")");
q_dot_qrs = strcat('sigma*epsilon*q_i+1',q_dot_qrs ); % append the coupling term

for i = 1:dim
    temp_q_dot = assign_index(q_dot_qrs, i, dim);
    if i < dim
    coupled_q_dot = strcat(coupled_q_dot, temp_q_dot,',');
    else
    coupled_q_dot = strcat(coupled_q_dot, temp_q_dot);
    end
end


full_string = strcat('par:sigma, beta, rho, epsilon;time:t;var:',...
    coupled_variables_x, ',',coupled_variables_q,',p',';fun:',coupled_lorenz,',',...
    coupled_q_dot,',',coupled_W,';');



function string_out = assign_index(string_in, i, dim)
if i < dim
    temp_string = replace(string_in, '_i+1', int2str(i+1));
else
    temp_string = replace(string_in, '_i+1', int2str(1));
end
string_out = replace(temp_string, '_i', int2str(i));
    
end