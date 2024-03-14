% nODE_validation
%
% Author: Elena Queirolo,
% Creation date: 27 Apr 2023
%
% This code uses totallygeneral_RHS_Hopf for trained RNN with a single layers
%

addpath '/Users/queirolo/Desktop/Research/packages/MatCont7p3'
init();

disp('Example: layer tanh with trained asymRNN')

validation = 1;

load('coefs_convolutionalNode.mat')
W = double(W);

dim = size(W,1);
R1 = triu(W,1);
diag_pert = diag(W) - gamma;
W = W - diag(diag(W));
R2 = W - R1 + R1.';
R2 = R2 + diag(diag_pert);

R2 = R2 * 10;

if dim < 30
    bifurcation_diag = 1;
else
    bifurcation_diag = 0;
end

linearfunc = cell(length(dim), 1);
linearfunc{1} = multilayer_lin_struct_asymRNN(dim, R1, R2);

nonlin = tanhnonlin();

out = allders(linearfunc, nonlin, dim);
rightHandSide = out{2};

[solutions, ~, ~, ~, ~, unproven] = totallygeneral_RHS_Hopf(dim(1), out, validation);


if bifurcation_diag
    bif_diag = figure;
    title('Bifurcation diagram')
    set(gca,'FontSize',16)
    xlabel('parameter','interpreter','latex','FontSize',21)
    ylabel('amplitude','interpreter','latex','FontSize',21)
    hold on
    all_pars = solutions(:,1);
    all_validated_pars = all_pars; all_validated_pars(unproven) = [];
    plot(all_validated_pars, 0*all_validated_pars, '*', 'MarkerSize', 4);
end

for index = 1:size(solutions,1)*bifurcation_diag
    if any(unproven == index)
        continue
    end
    
    [xHopf, pHopf] = extractSol(solutions,index, dim);
    fprintf('\n\nMatcont continuation at the Hopf bifurcation at parameter %f\n', pHopf)
    opt=contset;opt=contset(opt,'Singularities',1);
    opt = contset(opt,'MaxNumPoints',30);
    [x0,v0]=init_EP_EP(@() all_ders(linearfunc, nonlin, dim),xHopf.',pHopf-10^-4,[1]);
    
    [x,v,s,h,f]=cont(@equilibrium,x0,[],opt);
    
    x1=x(1:dim,s(2).index);
    par=x(end,s(2).index);
    
    opt = contset(opt,'MaxNumPoints',300);
    [x0,v0]=init_H_LC(@() all_ders(linearfunc, nonlin, dim),x1,par,[1],1e-3,20,4);
    
    [xlc,vlc,slc,hlc,flc]=cont(@limitcycle,x0,v0,opt);
    
    f = figure;
    axes
    plotcycle(xlc,vlc,slc,[size(xlc,1) 1 2]);
    %hold on
    
    figure(bif_diag)
    hold on
    M=max(xlc(1:dim:end-2,:));
    plot(xlc(end,:),M, 'LineWidth',2)
    xlim auto; ylim auto;
    xlim([0.297, 0.302])
    saveas(bif_diag, 'bifurcation_diag', 'epsc')
end
saveas(f, 'example_branch', 'epsc')
saveas(bif_diag, 'bifurcation_diag', 'epsc')



% _______________________________
%
%       FUNCTION DEFINITIONS
% _______________________________


% % % NONLINEARITIES

% hyperbolic tangent with derivatives
function nonlin = tanhnonlin()
nonlin = @tanh_nonlin;
    function [nonlin, d_nonlin, dd_nonlin, ddd_nonlin] = tanh_nonlin()
        
        nonlin = @(x) tanh(x);
        d_nonlin = @(x) 1-tanh(x).^2;
        dd_nonlin = @(x) - 2*tanh(x).*(1-tanh(x).^2);
        ddd_nonlin = @(x) - 2 + 8 * tanh(x).^2 - 6 * tanh(x).^4;
    end
end


% % % LINEAR PARTS

% multylayer AsymmetricRNN with respect to the identity shift
function linearfunc = multilayer_lin_struct_asymRNN(dim, R1, R2)
linearfunc = @linear_func;
    function [W, d_par_W] = linear_func
        W = @(a) (R1 - R1.' + a * eye(dim) + R2);
        d_par_W = @(a) eye(dim);
    end

end



% combine derivatives with full dimensionality - output compatible with
% MatCont
function out = all_ders(W, sigma, dims)
% gives derivatives of Sigma composed W
%
% Sigma returns
out = allders(W, sigma, dims);
out{7} = [];
end


% extract solutions from data given by general_RHS_Hopf
function [x,p] = extractSol(sol, index, dim)
if nargin < 2 || isempty(index)
    index = 1;
end
% each row of sol is: par, eigenval_imag, x, eigenvec_real, eigenvec_imag
p = sol(index, 1);
x = sol(index, 2 +(1:dim));
end
