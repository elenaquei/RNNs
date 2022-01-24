% validate branches of periodic orbits coming from the Hopf bifurcations

% uses "placeholder" library
% addpath('../bubbles/code ODE + simplices')
% start_placeholder

seed = 72; % to hange to get different W matrices
dim = 3;
[R1, R2,solutions, positive_lyap_index, negative_lyap_index, ...
    positive_lyap, negative_lyap, unproven] = asym_RHS_Hopf...
    (dim, 'perturbation', 0.1, 'seed', seed, 'validation', 1);

W = R1 - R1.' + R2;
% disp(W)
vectorfield = W_to_string(W);
% fprintf(s)
% works

n_non_comp_eqs = 3;
f_non_comp = @(x) non_computable_RNN(x);

for i = 1:1%size(solutions,1)
    X = solutions(i,:);
    [alpha_star, eigenval_imag, x_star, eigenvec_real, eigenvec_imag] = split_into_elements(X, dim);
    eigenvec = eigenvec_real + 1i*eigenvec_imag;
    n_nodes = 5; % number of Fourier nodes used: small, since near the Hopf bifurcation is a circle
    n_iter = 4; % number of iterations
    step_size = 10^-3; % initial step size (then adapted along the validation
    s = 'Hopf_RNN'; % where the solutions are stored
    
    f_RNN = from_string_to_polynomial_coef(vectorfield); % transformation into a vectorfield that can be used
    
    bool_saddle = 0;
    index_saddle = 2;
    if any(positive_lyap_index==i)
        sign_FLC = 1;
    else
        sign_FLC = -1;
    end
    % starting the continuation
    [sol_2, big_Hopf, x_dot_0] = Hopf_system_setup (alpha_star, x_star, f_RNN, n_nodes,...
        eigenvec, 1i*eigenval_imag, sign_FLC, step_size);
    
    % calling continuation with the Hopf boolean (guarantees that we keep using
    % the same scalar conditions over the continuation)
    bool_Hopf = 1;
    bool_fancy_scalar = 0; % already included in the Hopf boolean
    
    use_intlab = temp_intval;
    
    [s, x_n] = continuation ( sol_2, big_Hopf, n_iter, h, x_dot_0,s, 10^-6, ...
        bool_Hopf, bool_fancy_scalar, bool_saddle, index_saddle);
    
    figure('DefaultAxesFontSize',14)
    plot(last_sol, 'b', 'LineWidth',2)
    for j = 1:4
        subplot(1,4,j);
        hold on
    end
    plot(first_sol, 'g', 'LineWidth',2)
    for j = 1:4
        subplot(1,4,j);
        hold on
        ylim auto
        lim_y = ylim;
        if ~any(linspace(lim_y(1),...
                lim_y(2),6)==0)
            yTick_float = sort([linspace(lim_y(1),...
                lim_y(2),6),0]);
        else
            yTick_float = linspace(lim_y(1),...
                lim_y(2),6);
        end
        set(gca,'YTick',yTick_float)
        set(gca,'YTickLabel',sprintf('%1.2f|',yTick_float))
        xlabel('t','Interpreter','Latex')
        ylabel(sprintf('$u_%i$',j),'Interpreter','Latex')
    end
    set(gcf,'position',[100,200,900,400])
    disp('initial solution in green, end solution in blu')
    
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



function s = W_to_string(W)
% takes as input a square matrix W and returns the string corresponding to
% dot y = z
% dot z_i  = ( 1 - z_i^2) (Wy)_i    i = 1:dim

dim = size(W,1);
s = '';
for i = 1: dim    
    s = append(s, '- dot x', num2str(i), '+x', num2str(dim+i),'+l', num2str(1+i), '\n'); 
    % ls are extra variables, they must be =0, we need them to be able to
    % add the extra scalar equations
end
for i = 1:dim
    linear_string = '';
    minus_z2_linear_string = '';
    for j = 1:dim
        if W(i,j) > 0
            sign_ij = '+';
            minus_sign_ij = '-';
        else
            sign_ij = '-';
            minus_sign_ij = '+';
        end
        ij_abs_term = append(num2str(abs(W(i,j))), 'x', num2str(j));
        linear_string = append(linear_string,sign_ij, ij_abs_term);
        minus_z2_linear_string = append(minus_z2_linear_string,minus_sign_ij, ij_abs_term, 'x', num2str(dim+i),'^2');
        if i==j
            diag_elem = append('-l1 x',num2str(j));
            linear_string = append(linear_string,diag_elem);
            diag_elem_z2 = append('+l1 x',num2str(j),'x',num2str(dim+j),'^2');
            minus_z2_linear_string = append(minus_z2_linear_string,diag_elem_z2);
        end
    end
    s = append(s, '- dot x', num2str(dim+i), linear_string, minus_z2_linear_string,'\n');
end
end

function [F, dxF, dxF_mat, dxxF_norm] = non_computable_RNN(xi, size_scalar, W)

if isa(xi, 'Xi_vector')
    sum_x = zeros(xi.size_vector + xi.size_scalar,1);
    if isintval(xi) || use_intlab
        sum_x = intval(sum_x);
    end
    for j = 1: xi.size_scalar
        sum_x(j) = xi.scalar(j);
    end
    for j = 1:xi.size_vector
        sum_x(xi.size_scalar + j) = sum(xi.vector(j,:));
    end
else
    sum_x = xi;
    if isintval(xi) || use_intlab
        sum_x = intval(sum_x);
    end
end

x = sum_x(size_scalar+1:end);
size_vector = length(x);
size_y = size_vector/2;
y = x(1:size_y);
z = x(size_y+1:end);

F = - z + tanh(W*y);
if nargout <2
    return
end

gamma = sum_x(2);
W_gamma = W - eye(size_y)*gamma;

d_par_F = zeros(length(z), size_scalar);


d_par_F(:,2) = -diag(1-tanh(W_gamma*y).^2)*y;
d_z_F = -eye(size_y);
d_y_F = diag(1-tanh(W_gamma*y).^2) * W_gamma;
dxF = [d_par_F, d_y_F, d_z_F];
if nargout<3
    return
end

if isa(xi, 'Xivector')
    x_one_vec = 1 + 0*xi.vector(1,:);
    
    dxF_mat = [dxF(:,1:size_scalar), kron(dxF(:,(size_scalar+1):end), x_one_vec)];
else
    dxF_mat = NaN;
end
if nargout == 3
    return
end

% d_alphaalpha_F = -2*diag(tanh(W_gamma*y))*diag(1-tanh(W_gamma*y));
abs_Wy = abs(W_gamma)*abs(y);
d_alphay_F = diag(2*tanh(abs_Wy).*(1+tanh(abs_Wy)).*abs_Wy);
d_alphaalpha_F = 2 * tanh(abs_Wy).*(1+tanh(abs_Wy)).*abs(y).*abs(y);
one_vec = 0*y + 1;
d_yy_F = (-2*diag(tanh(abs_Wy))*diag(1+tanh(abs_Wy))*abs(W_gamma)) * abs(W_gamma)*one_vec;

dxxF_norm = zeros(length(sum_x), length(sum_x));
index_alpha = 2;
index_y = size_scalar+(1:size_y);
dxxF_norm(index_alpha,index_alpha) = d_alphaalpha_F;
dxxF_norm(index_alpha, index_y) = d_alphay_F(:);
dxxF_norm(index_y, index_alpha) = d_alphay_F(:);
dxxF_norm(index_y, index_y) = d_yy_F;

end