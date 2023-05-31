function out = allders(W, sigma, dims)
% INPUTS
% W         cell of any lenght, each element giving [W_f, d_par_W] = W{i}();
% sigma     elementwise nonlinearity, [sigma_f, d_x_sigma, d_xx_sigma, d_xxx_sigma] = sigma()
% dims      layer dimensions
%
% MODEL
% f(x) = sigma(Wn * sigma(Wn-1 * ... sigma( W1 * x))...)
%
% OUTPUT
% out
% with    out{1} = @init;
%         out{2} = @fun_eval;
%         out{3} = @jacobian;
%         out{4} = @jacobianp;
%         out{5} = @hessians;
%         out{6} = @hessiansp;
%         out{7} = @thirdderVW;

n_layers = length(W);
if n_layers ~= length(dims)
    error('Incompatible dimensions')
end

out = setup(dims(1));

for i = 1:n_layers
    out = recursive_all_ders(out, sigma, W{i}, dims(1));
end

end

function out = setup(dim) % IDENTITY
out{1} = @init;
out{2} = @fun_eval;
out{3} = @jacobian;
out{4} = @jacobianp;
out{5} = @hessians;
out{6} = @hessiansp;
out{7} = @thirdderVW;
out{8} = [];
out{9} = [];
    function dydt = fun_eval(t,x,par)
        if nargin == 2
            par = x;
            x = t;
            t = 0;
        end
        dydt = x;
    end
% --------------------------------------------------------------------------
    function [tspan,y0,options] = init
        y0=zeros(dim,1);
        handles = feval(setup(dim));
        options = odeset('Jacobian',handles(3),'JacobianP',handles(4),'Hessians',handles(5),'HessiansP',handles(6));
        tspan = [0 10];
    end
% --------------------------------------------------------------------------
    function jac = jacobian(t,x, par)
        
        jac= eye(dim, dim);
    end
% --------------------------------------------------------------------------
    function jacp = jacobianp(t,x,par) % dp
        
        jacp = zeros(dim,1);
    end
% --------------------------------------------------------------------------
    function hess = hessians(t,x,par)
        
        hess = zeros(dim, dim, dim);
    end
% --------------------------------------------------------------------------
    function hessp = hessiansp(t,x, par) % dx dp
        
        hessp = zeros(dim,dim);
    end
    function hessp = thirdderVW(t,x, par, v, w)
        
        hessp = zeros(dim,dim);
    end
end



function out = recursive_all_ders(in, sigma, W, dim_in)
% gives derivatives of Sigma composed W
%
% Sigma returns

[sigma_f, d_x_sigma, d_xx_sigma, d_xxx_sigma] = sigma();
[W_f, d_par_W] = W();

in_f = in{2};
d_x_in = in{3};
d_par_in = in{4};
d_xx_in = in{5};
d_xpar_in = in{6};
d_xxx_in_vw = in{7};

out{1} = @init;
out{2} = @fun_eval;
out{3} = @jacobian;
out{4} = @jacobianp;
out{5} = @hessians;
out{6} = @hessiansp;
out{7} = @thirdder;
out{8} = [];
out{9} = [];

    function dydt = fun_eval(t,x,par)
        if nargin == 2
            par = x;
            x = t;
            t = 0;
        end
        dydt = sigma_f(W_f(par)*in_f(x,par));
    end
% --------------------------------------------------------------------------
    function [tspan,y0,options] = init
        y0=zeros(dim_in,1);
        handles = feval(all_ders(sigma,W));
        options = odeset('Jacobian',handles(3),'JacobianP',handles(4),'Hessians',handles(5),'HessiansP',handles(6));
        tspan = [0 10];
    end
% --------------------------------------------------------------------------
    function jac = jacobian(t,x, par)
        if nargin == 2
            par = x;
            x = t;
            t = 0;
        end
        h = in_f(x,par);
        dh = d_x_in(x,par);
        
        jac= diag(d_x_sigma(W_f(par)*h)) * W_f(par)*dh; %new
    end
% --------------------------------------------------------------------------
    function jacp = jacobianp(t,x,par)
        if nargin == 2
            par = x;
            x = t;
            t = 0;
        end
        jacp = diag(d_x_sigma(W_f(par)*in_f(x,par))) *  ( d_par_W(par)*in_f(x,par) + W_f(par)*d_par_in(x,par)); 
    end
% --------------------------------------------------------------------------
    function hess = hessians(t,x,par)
        if nargin == 2
            par = x;
            x = t;
            t = 0;
        end
        
        Wh = W_f(par)*in_f(x,par); % vec length k
        
        JsigmaW = diag(d_x_sigma(Wh)) * W_f(par); % mat size k x m
        Hin = d_xx_in(x, par); % tens size m x n x n
        % first term JsigmaW * Hin
        
        Hsigma = d_xx_sigma(Wh); % tens k x k x k,  with only one diagonal non-zero!
        WJh = W_f(par)*d_x_in(x, par); % mat k x n
        
        % dimensions
        [k, m] = size(JsigmaW);
        n = size(WJh,2);
        % test other sizes
        if k~=length(Wh) || k~=size(Wh,1)  ...
                || k~=size(WJh,1) || m~=size(Hin,1)|| n~=size(Hin,2) || n~=size(Hin,3)
            error('Some sizes are incompatible')
        end
        
        hess = zeros(k,n,n);
        if isintval(x) || isintval(par)
            hess = intval(hess);
        end
        
        for i = 1:k
            for j = 1:n
                for l = 1:n
                    first_term = sum(squeeze(JsigmaW(i,:)).'.*squeeze(Hin(:,j,l)));
                    second_term_ijl = 0;
                    % formally, for-loop on k1 and Hsigma(i,k1,:) but only 
                    % Hsigma(i,i,i) is non-zero
                    % the full term
                    % sum(squeeze(Hsigma(i,k1,:)).*WJh(k1,j).*squeeze(WJh(:,l)))
                    % becomes
                    % sum(squeeze(Hsigma(i,i,i)).*WJh(i,j).*squeeze(WJh(i,l)))
                    % and thus, accounting for storage sizes
                    second_term_ijl = second_term_ijl + sum(squeeze(Hsigma(i)).*WJh(i,j).*squeeze(WJh(i,l)));
                    hess(i,j,l) = first_term + second_term_ijl;
                end
            end
        end
        
    end
% --------------------------------------------------------------------------
    function hessp = hessiansp(t,x, par)
        
        if nargin == 2
            par = x;
            x = t;
            t = 0;
        end
        dim = length(x);
        
        Jsigma = diag(d_x_sigma( W_f(par)*in_f(x,par)));
        JpW = d_par_W(par);
        first_term = Jsigma * (JpW*d_x_in(x,par) + W_f(par)*d_xpar_in(x,par));
        
        Hsigma = d_xx_sigma(W_f(par)*in_f(x,par));
        vec = d_par_W(par) * in_f(x,par) + W_f(par) * d_par_in(x,par);
        
        Hsigma_JpW = diag(Hsigma .* vec);
        
        hessp = first_term + Hsigma_JpW * W_f(par)*d_x_in(x,par);
        
    end
% --------------------------------------------------------------------------
    function third_der_dir = thirdder(t,x, par, v, w)
        if nargin == 4
            w = v;
            v = w;
            par = x;
            x = t;
            t = 0;
        end
        
        in_F = in_f(x,par);
        Wf = W_f(par);
        Wh = Wf*in_F;
        dim_out = length(Wh);
        
        Wdh = Wf*d_x_in(x,par); %
        Dxxx_sigma =  d_xxx_sigma(Wh);
        Dxx_sigma =  d_xx_sigma(Wh);
        D_sigma = d_x_sigma(Wh);
        Wdhv = Wdh * v;
        Wdhw = Wdh * w;
        Dxxh = d_xx_in(x,par);
        Wddhv = zeros(dim_out, dim_in);
        Wddhw = zeros(dim_out, dim_in);
        third_der_dir = zeros(dim_out,dim_in); % d3g Wdhv Wdhw Wdh
        if isintval(x) || isintval(par) || isintval(Wdh)
            
        third_der_dir = intval(third_der_dir); % d3g Wdhv Wdhw Wdh
            Wddhv = intval(Wddhv);
            Wddhw = intval(Wddhw);
        end
        
        Wdddhvw = Wf * d_xxx_in_vw(x, par,v,w);
        for i = 1:dim_out
            term_second_der = zeros(dim_in,1);
            for j = 1:dim_in
                for k = 1:dim_in
                    new_term = sum(squeeze(Wf(i,:)).' .* squeeze(Dxxh(:,j,k)));
                    Wddhv(i,k) = Wddhv(i,k) + new_term * v(j) + new_term * w(j);
                end
                
                term_second_der = term_second_der + squeeze(Dxxh(:,i,j)).*v(i).*w(j);
            end
        end
        Wddhvw = Wf * term_second_der;
        dim_out = size(Dxxx_sigma,1);
        
        for k = 1:dim_out
            for  l = 1:dim_in
                %for i = 1:dim_out % possible because sigma acts element wise!
                i = k;
                third_der_dir(k,l) = third_der_dir(k,l) + Dxxx_sigma(k)* Wdhv(i)* Wdhw(i) * Wdh(i,l)+ ...
                        Dxx_sigma(k) * Wdhv(i)* Wddhw(i,l) + Dxx_sigma(k) * Wdhw(i)* Wddhv(i,l)+ ...
                        Dxx_sigma(k) * Wdh(l)* Wddhvw(i) + D_sigma(k) * Wdddhvw(i,l);
                %end
            end
        end
        
    end
end