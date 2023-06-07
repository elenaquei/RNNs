function out = test_ODE
out{1} = @init;
out{2} = @fun_eval;
out{3} = @jacobian;
out{4} = @jacobianp;
out{5} = @hessians;
out{6} = @hessiansp;
out{7} = [];
out{8} = [];
out{9} = [];

dim = 6;
seed = 80;
rng('default')
rng(seed)
perturbation = 0.1;
R1 = randn(dim, dim);
R2 = perturbation * randn(dim, dim);

sigma_f = @(x) tanh(x); % vector in R^dim
d_x_sigma = @(x) diag(1-tanh(x).^2); % square matrix in R^dimx dim
d_xx_sigma = @(x) diagonal_tensor(- 2*tanh(x).*(1-tanh(x).^2)); % tensor in R^dim x dim x dim

W_f = @(x, a) (R1 - R1.' + a * eye(dim) + R2) * x;
d_x_W = @(x, a) R1 - R1.' + a * eye(dim) + R2;
d_par_W = @(x, a) x;
d_xpar_W = @(x, a) eye(dim, dim);
d_xx_W = @(x, a) zeros(dim, dim, dim);

% helper
    function tens = diagonal_tensor(vec)
        tens = zeros(dim, dim, dim);
        for i = 1:dim
            tens(i,i,i)= vec(i);
        end
    end

    function dydt = fun_eval(t,x,par)
        dydt = sigma_f(W_f(x,par));
    end
% --------------------------------------------------------------------------
    function [tspan,y0,options] = init
        y0=zeros(dim,1);
        handles = feval(test_ODE);
        options = odeset('Jacobian',handles(3),'JacobianP',handles(4),'Hessians',handles(5),'HessiansP',handles(6));
        tspan = [0 10];
    end
% --------------------------------------------------------------------------
    function jac = jacobian(t,x, par)
        jac= d_x_sigma(W_f(x,par))*d_x_W(x,par);
    end
% --------------------------------------------------------------------------
    function jacp = jacobianp(t,x,par)
        jacp = d_x_sigma(W_f(x,par)) * d_par_W(x,par);
    end
% --------------------------------------------------------------------------
    function hess = hessians(t,x,par)
        Jw = d_x_W(x,par);
        Hsigma = d_xx_sigma(W_f(x,par));
        Jsigma = d_x_sigma(W_f(x,par));
        Hw = d_xx_W(x,par);
        
        hess = zeros(dim,dim,dim);
        
        for j = 1:dim
            for l = 1:dim
                for m = 1:dim
                    second_term_jlm = 0;
                    for k1 = 1:dim
                        second_term_jlm = second_term_jlm + sum(squeeze(Hsigma(j,k1,:)).*Jw(k1,l).*squeeze(Jw(:,m)));
                    end
                    hess(j,l,m) = sum(Jsigma(j,:).*squeeze(Hw(:,l,m)).') + second_term_jlm;
                end
            end
        end
        
    end
% --------------------------------------------------------------------------
    function hessp = hessiansp(t,x, par)
        
        Jw = d_x_W(x,par);
        Jsigma = d_x_sigma(W(x,par));
        HpW = d_xpar_W(x,par);
        Hsigma = d_xx_sigma(x,par);
        JpW = d_par_W(x,par);
        
        Hsigma_JpW = zeros(dim,dim);
        for i = 1:dim
            for j = 1:dim
                Hsigma_JpW(i,j) = sum(Hsigma(i,j,:) .* JpW(:));
            end
        end
        
        
        hessp = Jsigma * HpW + Hsigma_JpW * Jw;
        
    end
end