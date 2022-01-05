% figure generation for paper "PLACEHOLDER',  E. Queirolo, C. Kuehn

asym_RHS_Hopf(6, 'perturbation', 0.1, 'seed', 80, 'validation', 0);

asym_RHS_Hopf(50, 'perturbation', 0.1, 'seed', 80, 'validation', 0);

asym_RHS_Hopf(150, 'perturbation', 0.1, 'seed', 80, 'validation', 0);

asym_RHS_Hopf(400, 'perturbation', 0.1, 'seed', 80, 'validation', 0);

asym_RHS_Hopf(50, 'perturbation', 0.1, 'seed', 120, 'validation', 0);

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
figure;
ndim=4;
M=max(x(1:ndim:end-2,:));
plot(x(end,:),M, 'LineWidth',2)
hold on
load('H_LC(2)_small.mat')
M=max(x(1:ndim:end-2,:));
plot(x(end,:),M, 'LineWidth',2)
set(gca,'FontSize',12)
xlabel('${\gamma}$','interpreter','latex')
ylabel('amplitude','interpreter','latex')



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

ndim=20;
figure;
% should be a loop on 10 elements, but some are repeated
for j = 1:13
    string = sprintf('H_LC(%s)_big.mat',num2str(j));
    load(string)
    M=max(x(1:ndim:end-2,:));
    plot(x(end,:),M, 'LineWidth',2)
    hold on
end

set(gca,'FontSize',12)
xlabel('${\gamma}$','interpreter','latex')
ylabel('amplitude','interpreter','latex')
