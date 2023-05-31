% multilayerRNN_Hopf_validation
% 
% Author: Elena Queirolo, 
% Creation date: 27 Apr 2023
% 
% This code tests genral_RHS_Hopf for RNN with multiple layers
% 

function multilayerRNN_Hopf_validation(dim, varargin)

if nargin == 0
    dim = [6]; %for antysimmetry, we need same dimensions
end
n_layers = length(dim);

R1 = cell(n_layers,1);
R2 = cell(n_layers,1);

for iter =1:n_layers
    [~, R1_iter, R2_iter] = input_parse(dim(iter), varargin{:}); % change this line to change the system
    R1{iter} = R1_iter;
    R2{iter} = R2_iter;
end

% definition of the nonlinear and linear part in the functions below
disp('Example 1: tanh with asymRNN')
linearfunc = cell(length(dim), 1);
for iter = 1:length(dim)
    linearfunc{iter} = multilayer_lin_struct_asymRNN(dim(iter), R1{iter}, R2{iter});
end
% testing sizes
% f1 = linearfunc{1};[a,b]= f1();
% f2 = linearfunc{2};

nonlin = tanhnonlin();
out = allders(linearfunc, nonlin, dim);
% testing allders
% out1 = out{1};
% out2 = out{2};
% out3 = out{3};
% out4 = out{4};
% out5 = out{5};
% out6 = out{6};
% out7 = out{7};


[sol_tanh_asym, ~, ~, ~, ~, unproven] = totallygeneral_RHS_Hopf(dim(1), out);

% % definition of the nonlinear and linear part in the functions below
% disp('Example 2: sin with asymRNN')
% linearfunc = lin_struct_asymRNN(dim, R1, R2);
% nonlin = sinnonlin();
% general_RHS_Hopf(dim, nonlin, linearfunc);
% 
% 
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






% DERIVATIVES TEST
% [xHopf, pHopf] = extractSol(sol_tanh_asym,2);
linearfunc_nodir = lin_struct_asymRNN_nodir(dim, R1, R2);
% [lin, d_x_lin, d_par_lin, d_xpar_lin, d_xx_lin] = linearfunc();
% feval(d_x_lin, xHopf.',pHopf)
% feval(lin, xHopf.',pHopf)
% 
% out = all_ders(tanhnonlin_nodir, linearfunc);
% jacobian = out{3};
% feval(jacobian, 0, xHopf', pHopf);


for index = 1:size(sol_tanh_asym,1)
    if any(unproven == index)
        continue
    end
    [xHopf, pHopf] = extractSol(sol_tanh_asym,index);
    opt=contset;opt=contset(opt,'Singularities',1);
    opt = contset(opt,'MaxNumPoints',30);
    [x0,v0]=init_EP_EP(@() all_ders(linearfunc, nonlin, dim),xHopf.',pHopf-.1,[1]);
    
    [x,v,s,h,f]=cont(@equilibrium,x0,[],opt);
    
    x1=x(1:dim,s(2).index);
    par=x(end,s(2).index);
    
    opt = contset(opt,'MaxNumPoints',1300);
    
    [x0,v0]=init_H_LC(@() all_ders(linearfunc, nonlin, dim),x1,par,[1],1e-3,20,4);
    
    opt = contset(opt,'Multipliers',1);
    opt = contset(opt,'Adapt',1);
    
    [xlc,vlc,slc,hlc,flc]=cont(@limitcycle,x0,v0,opt);
    
    figure
    axes
    plotcycle(xlc,vlc,slc,[size(xlc,1) 1 2]);
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



% ----- derivatives with full dimensionality (no dir vectors)
    function tens = diagonal_tensor(vec)
        tens = zeros(dim, dim, dim);
        for iter1 = 1:dim
            tens(iter1,iter1,iter1)= vec(iter1);
        end
    end

% hyperbolic tangent with derivatives
    function nonlin = tanhnonlin_nodir()
        nonlin = @tanh_nonlin;
        function [nonlin, d_nonlin, dd_nonlin] = tanh_nonlin()
            nonlin = @(x) tanh(x); % vector in R^dim
            d_nonlin = @(x) diag(1-tanh(x).^2); % square matrix in R^dimx dim
            dd_nonlin = @(x) diagonal_tensor(- 2*tanh(x).*(1-tanh(x).^2)); % tensor in R^dim x dim x dim
        end
    end


% sine with derivatives
    function nonlin = sinnonlin_nodir()
        nonlin = @sin_nonlin;
        function [nonlin, d_nonlin, dd_nonlin, ddd_nonlin] = sin_nonlin()
            
            nonlin = @(x) sin(x);
            d_nonlin = @(x) diag(cos(x));
            dd_nonlin = @(x) diagonal_tensor( - sin(x));
        end
    end



% % % LINEAR PARTS

% AsymmetricRNN with respect to the identity shift
    function linearfunc = lin_struct_asymRNN(dim, R1, R2)
        linearfunc = @linear_func;
        function [W, d_x_W, d_par_W, d_parpar_W, d_xpar_W, d_xx_W, d_xxx_W] = linear_func()
            W = @(x, a) (R1 - R1.' + a * eye(dim) + R2) * x;
            d_x_W = @(x, a) R1 - R1.' + a * eye(dim) + R2;
            d_par_W = @(x, a) x;
            d_parpar_W = @(x, y, a) zeros(dim,1);
            d_xpar_W = @(x, a) eye(dim, dim);
            d_xx_W = @(x, a, y) zeros(dim, dim);
            d_xxx_W = @(x, a, y, v) zeros(dim, dim);
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
    function linfunc = lin_struct_asymRNN_weights(dim, R1, R2, row, col)
        linfunc = @linear_func;
        function [W, d_x_W, d_par_W, d_parpar_W, d_xpar_W, d_xx_W, d_xxx_W] = linear_func()
            if nargin < 2 || isempty(col)
                col = floor(dim/3) +1;
            end
            if nargin < 1 || isempty(row)
                row = floor(dim/2)+1;
            end
            W = @(x, a) (R1 - R1.' + a * eye(dim) + R2) * x;
            d_x_W = @(x, a) R1 - R1.' + a * eye(dim) + R2;
            d_xpar_W = @(x, a) eye(dim, dim);
            d_xx_W = @(x, a, y) zeros(dim, dim);
            d_xxx_W = @(x, a, y, v) zeros(dim, dim);
            
            W_ij = zeros(dim,dim);
            W_ij(row, col) = 1;
            % if antisym, then also
            W_ij(col, row) = -1;
            d_par_W = @(x, par) W_ij * x;
            d_parpar_W = @(x, y, a) zeros(dim, 1);
        end
end

% generic RNN with respect to the identity shift
    function linfunc = lin_struct_RNN_weights(dim, R1, R2, row, col)
        linfunc = @linearfunc;
        function [W, d_x_W, d_par_W, d_parpar_W, d_xpar_W, d_xx_W, d_xxx_W] = linearfunc()
            if nargin < 2 || isempty(col)
                col = floor(dim/3) +1;
            end
            if nargin < 1 || isempty(row)
                row = floor(dim/2)+1;
            end
            W = @(x, a) (R1 - R1.' + a * eye(dim) + R2) * x;
            d_x_W = @(x, a) R1 - R1.' + a * eye(dim) + R2;
            d_xpar_W = @(x, a) eye(dim, dim);
            d_xx_W = @(x, a, y) zeros(dim, dim);
            d_xxx_W = @(x, a, y, v) zeros(dim, dim);
            
            W_ij = zeros(dim,dim);
            W_ij(row, col) = 1;
            % % if antisym, then also
            % W_ij(col, row) = -1;
            d_par_W = @(x, par) W_ij * x;
            d_parpar_W = @(x, y, a) zeros(dim, 1);
        end
    end


% ----- derivatives with full dimensionality (no dir vectors)

% AsymmetricRNN with respect to the identity shift
    function linearfunc = lin_struct_asymRNN_nodir(dim, R1, R2)
        linearfunc = @linear_func;
        function [W, d_x_W, d_par_W, d_xpar_W, d_xx_W] = linear_func()
            W = @(x, a) (R1 - R1.' + a * eye(dim) + R2) * x;
            d_x_W = @(x, a) R1 - R1.' + a * eye(dim) + R2;
            d_par_W = @(x, a) x;
            d_xpar_W = @(x, a) eye(dim, dim);
            d_xx_W = @(x, a) zeros(dim, dim, dim);
        end
    end

% AsymmetricRNN with respect to a weight
    function linfunc = lin_struct_asymRNN_weights_nodir(dim, R1, R2, row, col)
        linfunc = @linear_func;
        function [W, d_x_W, d_par_W, d_xpar_W, d_xx_W] = linear_func()
            if nargin < 2 || isempty(col)
                col = floor(dim/3) +1;
            end
            if nargin < 1 || isempty(row)
                row = floor(dim/2)+1;
            end
            W = @(x, a) (R1 - R1.' + a * eye(dim) + R2) * x;
            d_x_W = @(x, a) R1 - R1.' + a * eye(dim) + R2;
            d_xpar_W = @(x, a) eye(dim, dim);
            d_xx_W = @(x, a) zeros(dim, dim, dim);
            
            W_ij = zeros(dim,dim);
            W_ij(row, col) = 1;
            % if antisym, then also
            W_ij(col, row) = -1;
            d_par_W = @(x, par) W_ij * x;
        end
end

% generic RNN with respect to the identity shift
    function linfunc = lin_struct_RNN_weights_nodir(dim, R1, R2, row, col)
        linfunc = @linearfunc;
        function [W, d_x_W, d_par_W, d_xpar_W, d_xx_W] = linearfunc()
            if nargin < 2 || isempty(col)
                col = floor(dim/3) +1;
            end
            if nargin < 1 || isempty(row)
                row = floor(dim/2)+1;
            end
            W = @(x, a) (R1 - R1.' + a * eye(dim) + R2) * x;
            d_x_W = @(x, a) R1 - R1.' + a * eye(dim) + R2;
            d_xpar_W = @(x, a) eye(dim, dim);
            d_xx_W = @(x, a) zeros(dim, dim, dim);
            
            W_ij = zeros(dim,dim);
            W_ij(row, col) = 1;
            % % if antisym, then also
            % W_ij(col, row) = -1;
            d_par_W = @(x, par) W_ij * x;
            
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
%         [sigma_f, d_x_sigma, d_xx_sigma] = sigma();
%         [W_f, d_x_W, d_par_W, d_xpar_W, d_xx_W] = W();
%         
%         out{1} = @init;
%         out{2} = @fun_eval;
%         out{3} = @jacobian;
%         out{4} = @jacobianp;
%         out{5} = @hessians;
%         out{6} = @hessiansp;
%         out{7} = [];
%         out{8} = [];
%         out{9} = [];
%         function dydt = fun_eval(t,x,par)
%             dydt = sigma_f(W_f(x,par));
%         end
%         % --------------------------------------------------------------------------
%         function [tspan,y0,options] = init
%             y0=zeros(dim,1);
%             handles = feval(all_ders(sigma,W));
%             options = odeset('Jacobian',handles(3),'JacobianP',handles(4),'Hessians',handles(5),'HessiansP',handles(6));
%             tspan = [0 10];
%         end
%         % --------------------------------------------------------------------------
%         function jac = jacobian(t,x, par)
%             jac= d_x_sigma(W_f(x,par))*d_x_W(x,par);
%         end
%         % --------------------------------------------------------------------------
%         function jacp = jacobianp(t,x,par)
%             jacp = d_x_sigma(W_f(x,par)) * d_par_W(x,par);
%         end
%         % --------------------------------------------------------------------------
%         function hess = hessians(t,x,par)
%             Jw = d_x_W(x,par);
%             Hsigma = d_xx_sigma(W_f(x,par)); 
%             Jsigma = d_x_sigma(W_f(x,par));
%             Hw = d_xx_W(x,par);
%             
%             hess = zeros(dim,dim,dim);
%             
%             for j = 1:dim
%                 for l = 1:dim
%                     for m = 1:dim
%                         second_term_jlm = 0;
%                         for k1 = 1:dim
%                             second_term_jlm = second_term_jlm + sum(squeeze(Hsigma(j,k1,:)).*Jw(k1,l).*squeeze(Jw(:,m)));
%                         end
%                         hess(j,l,m) = sum(Jsigma(j,:).*squeeze(Hw(:,l,m)).') + second_term_jlm;
%                     end
%                 end
%             end
%             
%         end
%         % --------------------------------------------------------------------------
%         function hessp = hessiansp(t,x, par)
%             
%             Jw = d_x_W(x,par);
%             Jsigma = d_x_sigma(W_f(x,par));
%             HpW = d_xpar_W(x,par);
%             Hsigma = d_xx_sigma(W_f(x,par));
%             JpW = d_par_W(x,par);
%             
%             Hsigma_JpW = zeros(dim,dim);
%             for i = 1:dim
%                 for j = 1:dim
%                     Hsigma_JpW(i,j) = sum(squeeze(Hsigma(i,j,:)) .* JpW(:));
%                 end
%             end
%             
%             
%             hessp = Jsigma * HpW + Hsigma_JpW * Jw;
%             
%         end
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
        
        disp('remember to fix this')
        %rng('default')
        %rng(seed)
        
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

% extract solutions from data given by general_RHS_Hopf
    function [x,p] = extractSol(sol, index)
        if nargin < 2 || isempty(index)
            index = 1;
        end
        % each row of sol is: par, eigenval_imag, x, eigenvec_real, eigenvec_imag
        p = sol(index, 1);
        x = sol(index, 2 +(1:dim));
    end

end

