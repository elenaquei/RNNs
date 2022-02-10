function [fig_divergent, fig_transient] = visualisation_asym_RHS_Hopf(solutions, seed, R1, R2, index)

dim = (size(solutions,2)-2)/3;
W = @(a) R1 - R1.' + a * eye(dim) + R2;
f = @(x, a) asym_rhs(x, W(a));

% visualization : find last Hopf bifurcation
x = solutions(1:end,2+(1:dim));
bifurcation_values = solutions(1:end,1);
if nargin < 5
    [alpha_big, index] = max(bifurcation_values);
else
    alpha_big = bifurcation_values(index);
end
alpha_big = alpha_big + 10^-4;
fprintf('Parameter : %f\n',alpha_big)
eigenvec  = solutions(1:end,2+dim+(1:dim))+ 1i*solutions(1:end,2+2*dim+(1:dim));
eigenvec_plot = eigenvec(index,:);
plotting_dim = min(dim, 6);
% plot by finding the best orbit close to the expected periodic
% orbit generated by the last Hopf bifurcation
fig_divergent = figure;
[t,y] = ode45(@(t,x) f(x, alpha_big), [0,500], x(index,:) + sqrt(01000) * abs(eigenvec(index,:)));
plot(t, y(:,1:plotting_dim), 'LineWidth',3)
set(gca,'FontSize',18)
legend('$x_1$','$x_2$','$x_3$','$x_4$','$x_5$','$x_6$','Interpreter','latex')
ylabel('$x_i, \quad i=1,\dots, 6$','Interpreter','Latex')
xlabel('time','Interpreter','Latex')

fig_transient = figure;
[~,y] = ode45(@(t,x) -f(x, alpha_big), [0,150], x(index,:) + sqrt(0.10) * abs(eigenvec(index,:)));
[t,y] = ode45(@(t,x) f(x, alpha_big), [0,150], y(end,:));
plot(t, y(:,1:plotting_dim),'LineWidth',3)
set(gca,'FontSize',18)
legend('$x_1$','$x_2$','$x_3$','$x_4$','$x_5$','$x_6$','Interpreter','latex')
ylabel('$x_i,\quad i=1,\dots, 6$','Interpreter','Latex')
xlabel('time','Interpreter','Latex')
% plot_bifurcation_diag(f, x, bifurcation_values, eigenvec)


% special plots for cases presented in paper
if nargin> 2
    if dim == 6 && seed == 80
        fig_transient = figure;
        [~,y] = ode45(@(t,x) -f(x, alpha_big), [0,150], x(index,:) + sqrt(0.10) * abs(eigenvec(index,:)));
        [t,y] = ode45(@(t,x) f(x, alpha_big), [0,100], y(end,:));
        plot(t(300:end)-t(300), y(300:end,1:plotting_dim),'LineWidth',3)
        set(gca,'FontSize',18)
    elseif dim == 50 && seed == 80
        fig_transient = figure;
        [~,y] = ode45(@(t,x) -f(x, alpha_big), [0,350], x(index,:) + sqrt(0.10) * abs(eigenvec(index,:)));
        [t,y] = ode45(@(t,x) f(x, alpha_big), [0,45], y(end,:));
        plot(t, y(:,1:plotting_dim),'LineWidth',3)
        set(gca,'FontSize',18)
    elseif dim == 50 && seed == 120
        fig_transient = figure;
        [~,y] = ode45(@(t,x) -f(x, alpha_big), [0,900], x(index,:) + sqrt(0.10) * abs(eigenvec(index,:)));
        [t,y] = ode45(@(t,x) f(x, alpha_big), [0,100], y(end,:));
        plot(t, y(:,1:plotting_dim),'LineWidth',3)
        set(gca,'FontSize',18)
    elseif dim == 400 && seed == 80
        fig_transient = figure;
        [~,y] = ode45(@(t,x) -f(x, alpha_big), [0,150], x(index,:) + sqrt(0.10) * abs(eigenvec(index,:)));
        [t,y] = ode45(@(t,x) f(x, alpha_big), [0,20], y(end,:));
        plot(t, y(:,1:plotting_dim),'LineWidth',3)
        set(gca,'FontSize',18)
    end
end
legend('$x_1$','$x_2$','$x_3$','$x_4$','$x_5$','$x_6$','Interpreter','latex')
ylabel('$x_i,\quad i=1,\dots, 6$','Interpreter','Latex')
xlabel('time','Interpreter','Latex')
end


function h_dot = asym_rhs(h, W)
h_dot = tanh( W * h );
end