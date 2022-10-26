% main_genRNN_Hopf_validations
% 
% Author: Elena Queirolo, 
% Creation date: 26 Oct 2022
% 
% This code tests genral_RHS_Hopf of different RNNs, with and without
% antisymmetry.
% 

function main_genRNN_Hopf_validation()

% first example: AsymmetricRNN
% input parsing and definition of system parameters
dim = 6;
[dim, R1, R2, validation] = input_parse(dim); % change this line to change the system


% definition of the nonlinear and linear part in the functions below
linearfunc = lin_struct_asymRNN(dim, R1, R2);
nonlin = tanhnonlin();
general_RHS_Hopf(dim, nonlin, linearfunc)

% definition of the nonlinear and linear part in the functions below
linearfunc = lin_struct_asymRNN(dim, R1, R2);
nonlin = sinnonlin();
general_RHS_Hopf(dim, nonlin, linearfunc)


% second example: AsymmetricRNN w.r.t. a "Random" weight
dim = 6;
row = 2;
col = 6;
linearfunc = lin_struct_asymRNN_weights(dim, R1, R2, row, col);
nonlin = tanhnonlin();
general_RHS_Hopf(dim, nonlin, linearfunc)


% third example: AsymmetricRNN w.r.t. a "Random" weight
dim = 6;
row = 2;
col = 6;
linearfunc = lin_struct_RNN_weights(dim, R1, R2, row, col);
nonlin = tanhnonlin();
general_RHS_Hopf(dim, nonlin, linearfunc)



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





    function [dim, R1, R2, validation] = input_parse(dim, varargin)
        defaultPerturbation = 0.1;
        defaultSeed = 80;
        defaultValidation = 1;
        defaultR1 = zeros(dim, dim);
        defaultR2 = zeros(dim, dim);

        p = inputParser;
        validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
        validInteger = @(x) isnumeric(x) && isscalar(x) && (mod(x,1) ==0);
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

end