% figure generation for paper "PLACEHOLDER',  E. Queirolo, C. Kuehn
addpath '/Users/queirolo/Desktop/Research/packages/MatCont7p3'
init();


[R1, R2,sol]=asym_RHS_Hopf(6, 'perturbation', 0.1, 'seed', 80, 'validation', 1);
[fig_divergent, fig_transient] =visualisation_asym_RHS_Hopf(sol, 80, R1, R2);
saveas(fig_transient,'dim6_transient','epsc')
saveas(fig_divergent,'dim6_divergent','epsc')

[R1, R2,sol]=asym_RHS_Hopf(50, 'perturbation', 0.1, 'seed', 80, 'validation', 0);
[fig_divergent, fig_transient] =visualisation_asym_RHS_Hopf(sol, 80,R1, R2);
saveas(fig_transient,'dim50_transient','epsc')
saveas(fig_divergent,'dim50_divergent','epsc')

% [R1, R2,sol]=asym_RHS_Hopf(150, 'perturbation', 0.1, 'seed', 80, 'validation', 1);
% [fig_divergent, fig_transient] =visualisation_asym_RHS_Hopf(sol, 80,R1, R2);
% not in the paper

[R1, R2,sol]=asym_RHS_Hopf(400, 'perturbation', 0.1, 'seed', 80, 'validation', 0); % change to validation
[fig_divergent, fig_transient] =visualisation_asym_RHS_Hopf(sol, 80,R1, R2);
saveas(fig_transient,'dim400_transient','epsc')
saveas(fig_divergent,'dim400_divergent','epsc')

[R1, R2,sol, ~, ~, ~, ~, unproven]=asym_RHS_Hopf(50, 'perturbation', 0.1, 'seed', 120, 'validation', 1);
[fig_divergent, fig_transient] = visualisation_asym_RHS_Hopf(sol, 120,R1, R2, unproven(1));
saveas(fig_transient,'dim50_singular','epsc')

R1 =[0.0820    1.9805   -1.0459   -0.9195
    0.6341    0.7433   -0.9086    0.1274
   -0.4147   -1.6477    2.0508    0.7477
    0.7523   -0.6576   -1.0741   -1.4106];
R2 = [0.0929    0.0645   -0.0047    0.0236
   -0.0529   -0.0672    0.0852    0.0022
    0.1457   -0.0213    0.0325   -0.0131
   -0.2527   -0.0332    0.0977    0.0373];
W = R1 - R1.' + R2;
[~,~,sol]=asym_RHS_Hopf(4, 'R1', R1, 'R2', R2);

% this system is then analysed in MatCont, and the periodic orbits
% generated for the Hopf bifurcations are followed, thus giving
load('H_LC(1)_small.mat')
fig_4D = figure;
ndim=4;
M=max(x(1:ndim:end-2,:));
plot(x(end,:),M, 'LineWidth',2)
hold on
load('H_LC(2)_small.mat')
M=max(x(1:ndim:end-2,:));
plot(x(end,:),M, 'LineWidth',2)
set(gca,'FontSize',16)
xlabel('${\gamma}$','interpreter','latex','FontSize',21)
ylabel('amplitude','interpreter','latex','FontSize',21)

saveas(fig_4D,'bifurcation_diagram_4D','epsc')



[R1, R2,solutions, positive_lyap_index, negative_lyap_index,...
    positive_lyap, negative_lyap, unproven] = asym_RHS_Hopf(20, ...
    'perturbation', 0.1, 'seed', 80, 'validation', 0);
gammas = solutions(:,1);
figure; plot(sort(gammas))
W = R1 - R1.'+ R2;

[string_system, string_variables] = mat2ODEstr(W);
fprintf(string_system)
fprintf('\n\n')
% fprintf(string_variables)
% fprintf('\n\n')

% data from MatCont is saved, here only direct plotting
ndim=20;
fig_20D = figure;
for j = 1:10
    string = sprintf('H_LC(%s)_big.mat',num2str(j));
    load(string)
    M=max(x(1:ndim:end-2,:));
    plot(x(end,:),M, 'LineWidth',2)
    hold on
end

set(gca,'FontSize',16)
xlabel('${\gamma}$','interpreter','latex','FontSize',21)
ylabel('amplitude','interpreter','latex','FontSize',21)

saveas(fig_20D,'bifurcation_diagram_20D','epsc')

