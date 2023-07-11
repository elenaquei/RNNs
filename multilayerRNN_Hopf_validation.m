% multilayerRNN_Hopf_validation
%
% Author: Elena Queirolo,
% Creation date: 27 Apr 2023
%
% This code tests genral_RHS_Hopf for RNN with multiple layers
%

function multilayerRNN_Hopf_validation(dim, n_case, varargin)

if nargin == 0
    dim = [6, 6]; %for antysimmetry, we need same dimensions
end
n_layers = length(dim);

R1 = cell(n_layers,1);
R2 = cell(n_layers,1);

for iter =1:n_layers
    [~, R1_iter, R2_iter] = input_parse(dim(iter), varargin{:}); % change this line to change the system
    R1{iter} = R1_iter; %  toeplitz(R1_iter(1,:)); 
    R2{iter} = R2_iter;
end

switch n_case
    case 1
        % one layer, overwrites the weights for security
        disp('Example 1: 1 layer tanh with asymRNN')
        dim = dim(1);
        linearfunc = cell(length(dim), 1);
        for iter = 1:length(dim)
            linearfunc{iter} = multilayer_lin_struct_asymRNN(dim(iter), R1{iter}, R2{iter});
        end
        nonlin = tanhnonlin();
        % unit_test_1layer();
        
    case 2
        disp('Example 2: multilayer tanh with asymRNN')
        linearfunc = cell(length(dim), 1);
        for iter = 1:length(dim)
            linearfunc{iter} = multilayer_lin_struct_asymRNN(dim(iter), R1{iter}, R2{iter});
        end
        nonlin = tanhnonlin();
        % unit_test_2layers();
        
    case 3
        disp('Example 3: sin with asymRNN')
        linearfunc = cell(length(dim), 1);
        for iter = 1:length(dim)
            linearfunc{iter} = multilayer_lin_struct_asymRNN(dim(iter), R1{iter}, R2{iter});
        end
        nonlin = sinnonlin();
        
    case 4
        disp('Example 3: tanh with a weight in asymRNN')
        dim = 6;
        row = 6;
        col = 6;
        gamma = 10^-2;
        
        linearfunc = cell(length(dim), 1);
        for iter = 1:length(dim)
            linearfunc{iter} = lin_struct_asymRNN_weights(dim(iter), R1{iter}, R2{iter}, gamma, row, col);
        end
        nonlin = tanhnonlin();
    case 5
        disp('Example 5: tanh with a weight off-diag in asymRNN')
        dim = 6;
        row = 3;
        col = 6;
        gamma = 10^-2;
        
        linearfunc = cell(length(dim), 1);
        for iter = 1:length(dim)
            linearfunc{iter} = lin_struct_asymRNN_weights(dim(iter), R1{iter}, R2{iter}, gamma, row, col);
        end
        nonlin = tanhnonlin();
        
        end
        nonlin = tanhnonlin();
end

out = allders(linearfunc, nonlin, dim);

if n_case == 2
    %unit_test_2layers()
end

[sol_tanh_asym, ~, ~, ~, ~, unproven] = totallygeneral_RHS_Hopf(dim(1), out);

if n_case ==1
    nargout{3} = sol_tanh_asym;
end

% % second example: AsymmetricRNN w.r.t. a "Random" weight
% disp('Example 3: tanh with a weight in asymRNN')
% dim = 6;
% row = 2;
% col = 6;
% linearfunc = lin_struct_asymRNN_weights(dim, R1, R2, row, col);
% nonlin = tanhnonlin();
% general_RHS_Hopf(dim, nonlin, linearfunc);
% 
% 
% % third example: AsymmetricRNN w.r.t. a "Random" weight
% disp('Example 4: tanh with a wieght that breaks the symmetry in asymRNN')
% dim = 6;
% row = 2;
% col = 6;
% linearfunc = lin_struct_RNN_weights(dim, R1, R2, row, col);
% nonlin = tanhnonlin();
% general_RHS_Hopf(dim, nonlin, linearfunc);


bif_diag = figure;
title('Bifurcation diagram')
hold on

for index = 1:size(sol_tanh_asym,1)
    if any(unproven == index)
        continue
    end
    
    [xHopf, pHopf] = extractSol(sol_tanh_asym,index);
    fprintf('\n\nMatcont continuation at the Hopf bifurcation at parameter %f\n', pHopf)
    opt=contset;opt=contset(opt,'Singularities',1);
    opt = contset(opt,'MaxNumPoints',30);
    [x0,v0]=init_EP_EP(@() all_ders(linearfunc, nonlin, dim),xHopf.',pHopf-.1,[1]);
    
    [x,v,s,h,f]=cont(@equilibrium,x0,[],opt);
    
    x1=x(1:dim,s(2).index);
    par=x(end,s(2).index);
    
    opt = contset(opt,'MaxNumPoints',6300);
    [x0,v0]=init_H_LC(@() all_ders(linearfunc, nonlin, dim),x1,par,[1],1e-3,20,4);
    
    [xlc,vlc,slc,hlc,flc]=cont(@limitcycle,x0,v0,opt);
    
    figure
    axes
    plotcycle(xlc,vlc,slc,[size(xlc,1) 1 2]);
    %hold on
    
    figure(bif_diag)
    M=max(xlc(1:dim:end-2,:));
    plot(xlc(end,:),M, 'LineWidth',2)
end





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


% sine with derivatives
    function nonlin = sinnonlin()
        nonlin = @sin_nonlin;
        function [nonlin, d_nonlin, dd_nonlin, ddd_nonlin] = sin_nonlin()
            
            nonlin = @(x) sin(x);
            d_nonlin = @(x) cos(x);
            dd_nonlin = @(x) - sin(x);
            ddd_nonlin = @(x) - cos(x);
        end
    end


% % % LINEAR PARTS

% AsymmetricRNN with respect to the identity shift
    function linearfunc = lin_struct_asymRNN(dim, R1, R2)
        linearfunc = @linear_func;
        function [W, d_par_W] = linear_func()
            W = @(a) (R1 - R1.' + a * eye(dim) + R2) * x;
            d_par_W = @(a) eye(dim);
        end
    end

% multylayer AsymmetricRNN with respect to the identity shift
    function linearfunc = multilayer_lin_struct_asymRNN(dim, R1, R2)
        linearfunc = @linear_func;
        function [W, d_par_W] = linear_func
            W = @(a) (R1 - R1.' + a * eye(dim) + R2);
            d_par_W = @(a) eye(dim);
        end
        
    end


% AsymmetricRNN with respect to a weight
    function linfunc = lin_struct_asymRNN_weights(dim, R1, R2, gamma, row, col)
        linfunc = @linear_func;
        function [W, d_par_W] = linear_func()
            if nargin < 2 || isempty(col)
                col = floor(dim/3) +1;
            end
            if nargin < 1 || isempty(row)
                row = floor(dim/2)+1;
            end
            W_fixed = R1 - R1.' + gamma * eye(dim) + R2;
            
            W_ij = zeros(dim); 
            W_ij(row, col) = 1;
            W_ij(col, row) = -1;
            
            W = @(a) W_fixed + a * W_ij;
            d_par_W = @(par) W_ij;
        end
    end

% generic RNN with respect to the identity shift
    function linfunc = lin_struct_RNN_weights(dim, R1, R2, row, col)
        linfunc = @linearfunc;
        function [W, d_par_W] = linearfunc()
            if nargin < 2 || isempty(col)
                col = floor(dim/3) +1;
            end
            if nargin < 1 || isempty(row)
                row = floor(dim/2)+1;
            end
            W = @(a) (R1 - R1.' + a * eye(dim) + R2);
            
            W_ij = zeros(dim,dim);
            W_ij(row, col) = 1;
            % % if antisym, then also
            % W_ij(col, row) = -1;
            d_par_W = @(x, par) W_ij;
        end
    end


    function linfunc = lin_struct_random_weights(dim, R1, row, col)
        linfunc = @linearfunc;
        function [W, d_par_W] = linearfunc()
            if nargin < 2 || isempty(col)
                col = floor(dim/3) +1;
            end
            if nargin < 1 || isempty(row)
                row = floor(dim/2)+1;
            end
            W = @(a) R1;
            
            W_ij = zeros(dim,dim);
            W_ij(row, col) = 1;
            % % if antisym, then also
            % W_ij(col, row) = (col~=row)*-1;
            d_par_W = @(x, par) W_ij;
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
    function [x,p] = extractSol(sol, index)
        if nargin < 2 || isempty(index)
            index = 1;
        end
        % each row of sol is: par, eigenval_imag, x, eigenvec_real, eigenvec_imag
        p = sol(index, 1);
        x = sol(index, 2 +(1:dim));
    end



% INPUT PARSER

    function [dim, R1, R2, validation] = input_parse(dim, varargin)
        defaultPerturbation = 0.1;
        defaultSeed = 80;
        defaultValidation = 1;
        defaultR1 = zeros(dim, dim);
        defaultR2 = zeros(dim, dim);
        
        p = inputParser;
        validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
        validInteger = @(x) isnumeric(x) && isscalar(x) && (mod(x,1) ==0);
        addRequired(p,'dimension',validInteger);
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





% unit testing - to run if anything changes
    function unit_test_2layers()
        % testing sizes
        f1 = linearfunc{1};
        [a,b]= f1();
        if any(size(a(1)) ~= 6) || any(size(b(2)) ~= 6)
            error('Wrong sizes')
        end
        f2 = linearfunc{2};
        [a,b]= f2();
        if any(size(a(1)) ~= 6) || any(size(b(2)) ~= 6)
            error('Wrong sizes')
        end
        
        nonlin = tanhnonlin();
        out = allders(linearfunc, nonlin, dim);
        % testing allders
        out1 = out{1};
        out2 = out{2}; % fun eval
        out3 = out{3}; % DF
        out4 = out{4}; % DpF
        out5 = out{5}; % DxxF
        out6 = out{6}; % DxpF
        out7 = out{7}; % Dxxxfvw
        
        % multilayer test
        p = rand;
        x = rand(6,1);
        W1 = R1{1} - R1{1}.' + p * eye(6) + R2{1};
        W2 = R1{2} - R1{2}.' + p * eye(6) + R2{2};
        fxp = @(x) tanh( W2 * tanh(W1 * x));
        if norm(out2(x,p) - fxp(x))>10^-5
            error('Wrong function')
        end
        % multilayer derivative test
        DF_diff = (out3(x,p) - finite_diff_fn(fxp, x));
        if norm(DF_diff)>10^-3
            error('Wrong derivative')
        end
        
        % second der
        DDF_diff = (out5(x,p) - reshape(finite_diff_fn(@(x)reshape(out3(x,p),36,[]), x, 10^-6), 6,6,6));
        if norm(reshape(DDF_diff,1,[]))>10^-3
            error('Wrong second derivative')
        end
        
        W1 = @(p) R1{1} - R1{1}.' + p * eye(6) + R2{1};
        W2 = @(p) R1{2} - R1{2}.' + p * eye(6) + R2{2};
        fxp = @(p) tanh( W2(p) * tanh(W1(p) * x));
        
        % parameter derivative test
        DF_diff = (out4(x,p) - finite_diff_fn(fxp, p));
        if norm(DF_diff)>10^-3
            error('Wrong parameter derivative')
        end
        % mixed der
        DDpF_diff = (out6(x,p) - reshape(finite_diff_fn(@(p)reshape(out3(x,p),36,[]), p), 6,6));
        if norm(reshape(DDpF_diff,1,[]))>10^-3
            error('Wrong second derivative')
        end
        
        disp('Unit testing successful')
    end

% unit testing for single layer - sanity check
    function unit_test_1layer()
        % testing sizes
        f1 = linearfunc{1};
        [a,b]= f1();
        if any(size(a(1)) ~= 6) || any(size(b(2)) ~= 6)
            error('Wrong sizes')
        end
        
        nonlin = tanhnonlin();
        out = allders(linearfunc, nonlin, dim);
        % testing allders
        out1 = out{1};
        out2 = out{2}; % fun eval
        out3 = out{3}; % DF
        out4 = out{4}; % DpF
        out5 = out{5}; % DxxF
        out6 = out{6}; % DxpF
        out7 = out{7}; % Dxxxfvw
        
        % single layer function test
        p = rand;
        x = rand(6,1);
        W1 = R1{1} - R1{1}.' + p * eye(6) + R2{1};
        fxp = @(x) tanh(W1 * x);
        if norm(out2(x,p) - fxp(x))>10^-5
            error('Wrong function')
        end
        
        % single layer derivative test
        DF_diff = (out3(x,p) - finite_diff_fn(fxp, x));
        if norm(DF_diff)>10^-3
            error('Wrong derivative')
        end
        
        % second der
        DDF_diff = (out5(x,p) - reshape(finite_diff_fn(@(x)reshape(out3(x,p),36,[]), x), 6,6,6));
        if norm(reshape(DDF_diff,1,[]))>10^-3
            error('Wrong second derivative')
        end
        
        W1 = @(p) R1{1} - R1{1}.' + p * eye(6) + R2{1};
        fxp = @(p) tanh(W1(p) * x);
        
        % parameter derivative test
        DF_diff = (out4(x,p) - finite_diff_fn(fxp, p));
        if norm(DF_diff)>10^-3
            error('Wrong parameter derivative')
        end
        % mixed der
        DDpF_diff = (out6(x,p) - reshape(finite_diff_fn(@(p)reshape(out3(x,p),36,[]), p), 6,6));
        if norm(reshape(DDpF_diff,1,[]))>10^-3
            error('Wrong second derivative')
        end
        
        disp('Unit testing successful')
    end


end

