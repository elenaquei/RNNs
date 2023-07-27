% new figure generation with multiple layers

% figure generation for paper "Computer Validation of Neural Network
% Dynamics: A First Case Study",  E. Queirolo, C. Kuehn

% ADAPT TO OWN COMPUTER
addpath '/Users/queirolo/Desktop/MatCont7p3'
init();

save_fig = 1;

[rhs, sol]=multilayerRNN_Hopf_validation(6,1, 'perturbation', 0.1, 'seed', 80, 'validation', 1);
if save_fig
    saveas(gca,'dim6_bifdiag','epsc')
end
[fig_divergent, fig_transient] =visualisation_asym_RHS_Hopf(sol, 80, rhs);
if save_fig
    saveas(fig_transient,'dim6_transient','epsc')
    saveas(fig_divergent,'dim6_divergent','epsc')
end

[rhs, sol]=multilayerRNN_Hopf_validation(50, 1,'perturbation', 0.1, 'seed', 80, 'validation', 1, 'bifurcation', 0);
[fig_divergent, fig_transient] =visualisation_asym_RHS_Hopf(sol, 80, rhs);
if save_fig
    saveas(fig_transient,'dim50_transient','epsc')
    saveas(fig_divergent,'dim50_divergent','epsc')
end
% [R1, R2,sol]=asym_RHS_Hopf(150, 'perturbation', 0.1, 'seed', 80, 'validation', 1, 'bifurcation', 0);
% [fig_divergent, fig_transient] =visualisation_asym_RHS_Hopf(sol, 80, rhs);
% not in the paper

% ONLY ONCE
%[rhs, sol]=multilayerRNN_Hopf_validation(400, 1, 'perturbation', 0.1, 'seed', 80, 'validation', 1, 'bifurcation', 0); % change to validation
%[fig_divergent, fig_transient] =visualisation_asym_RHS_Hopf(sol, 80, rhs);
%saveas(fig_transient,'dim400_transient','epsc')
%saveas(fig_divergent,'dim400_divergent','epsc')

[rhs, sol, unproven]=multilayerRNN_Hopf_validation(50, 1,'perturbation', 0.1, 'seed', 120, 'validation', 1, 'bifurcation', 0);
[fig_divergent, fig_transient] = visualisation_asym_RHS_Hopf(sol, 120, rhs, unproven(1));
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
[~,sol]=multilayerRNN_Hopf_validation(4,1, 'R1', R1, 'R2', R2, 'validation', 1);
title('')
if save_fig
    saveas(gcf,'bifurcation_diagram_4D','epsc')
end

[rhs, solutions] = multilayerRNN_Hopf_validation(20, 1,...
    'perturbation', 0.1, 'seed', 80, 'validation', 1);
title('')
if save_fig
    saveas(gcf,'bifurcation_diagram_20D','epsc')
end

%%%%%%%%% multilayer

[rhs, sol]=multilayerRNN_Hopf_validation([6,6,6],2, 'perturbation', 0.1, 'seed', 80, 'validation', 1);
title('')
axis([-0.15, 0.05, 0, 1.25])
if save_fig
    saveas(gcf,'bifurcation_diagram_multilayer','epsc')
end

%%%%%%%%% 1 layer, parameter is a diagonal element

% only 1 Hopf found and validated
% [rhs, sol]=multilayerRNN_Hopf_validation(6,4, 'perturbation', 0.1, 'seed', 80, 'validation', 1);
% 3 validated Hopf
[rhs, sol]=multilayerRNN_Hopf_validation(6,4, 'perturbation', 0.1, 'seed', 90, 'validation', 1);
title('')
if save_fig
    saveas(gcf,'bifurcation_diagram_diag','epsc')
end
%%%%%%%%% 1 layer, parameter is an off-diagonal element

[rhs, sol]=multilayerRNN_Hopf_validation(6,5, 'perturbation', 0.1, 'seed', 80, 'validation', 1);
title('')
if save_fig
    saveas(gcf,'bifurcation_diagram_offdiag','epsc')
end
% not in paper:
%[rhs, sol]=multilayerRNN_Hopf_validation(6,5, 'perturbation', 0.1, 'seed', 180, 'validation', 1);

