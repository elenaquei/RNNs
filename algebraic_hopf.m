function [x_star,lambda_star,eigenvec,eigenval, l1] = ...
    algebraic_hopf(f,df,ddf,dddf,dalphaf, dalphaxf, dalphaxxf, dalphaalphaf,...
    dalphaalphaxf, N,X,phi,bool_val)
 
%function [x_star,lambda_star,eigenvec,eigenval] = ...
%    algebraic_hopf_lorenz(f,df,ddf,dddf,N,X,phi)
%
% code to find and algebraically validate a Hopf bifurcation in an ODE
% system
% Does not require the ODE to be polynomial
% All function handles have as input ()
% INPUT
% f     function handle, right hand side of the ODE
% df    function handle, first derivative of f
% ddf   function handle, second derivative of f,input x and the two
%       directions of the derivative
% dddf  function handle, third derivative of f, input x and three 
%       directions of the derivative
%       return directional derivatives
% N     integer, dimension of the system 
% X     2+3*N vector, approximation of the Hopf point (optional, DEFAULT
%       random)
% phi   2+3*N vector, (optional, DEFAULT random)
% OUTPUT (VALIDATED)
% x_star        x-position of Hopf bifurcation
% lambda_star   parameter value at the Hopf bifurcation
% eigenvec      eigenvector 
% eigenval      eigenvalue 
% l1            <0 supercritical, >0 subcritical Hopf bifurcation 

% supercritical : stable limit cycle
% subcritical : unstable limit cycle
if nargin<8
    bool_val = 1;
end
if nargin<7
    phi = rand(N,1);
end
if nargin<6
    X = rand(2+3*N,1);
elseif length(X)==2+N
    X = [X;rand(2*N,1)];
end
if nargin<5
    error('Too few input arguments')
end

F=F_general_Hopf(f,X,phi,df);

if norm(F)>10^-6
    [X]=newton_Hopf(f,X,phi);
end

alpha=X(1); % Parameter 
eigenval_imag=X(2); % distance of the complex conjugate eigs from the real axis

x=X(2+(1:N)); % the variables from the model
eigenvec_real=X(N+2+(1:N)); 
eigenvec_imag=X(2*N+2+(1:N)); % The real and imaginary parts of the eigenvector

%%%% validation
% for the validation, the exact derivative of f is necessary
DF = Df_Hopf(f, df, ddf, dalphaf, dalphaxf, phi, alpha, eigenval_imag, x, eigenvec_real, eigenvec_imag);
if rank(DF)-size(DF,1)<0
    error('Derivative not invertible')
end

A=inv(DF);
DF = intval(DF);
A = intval(A);

% start intlab
F=F_general_Hopf(f,intval(X),phi,df);
Y = norm(A*F);

Z1 = norm( eye(length(X)) - A *DF);

R = 0.3; % first approximation of the maximum radius

XandR=infsup(X-R,X+R);
DDF= second_derivative_F(XandR,ddf,dddf, dalphaxf, dalphaxxf, dalphaalphaf, dalphaalphaxf);

Z2 = norm(sup(abs(A*DDF)));

pol=@(r) Y+(Z1-1)*r+Z2*r.^2;

delta= (intval(Z1)-1)^2-4*intval(Z2)*intval(Y);
if delta<=0 
    error('Hopf not validated')
end
rmin=sup((-(Z1-1)-sqrt(delta))/(2*Z2));
rmax=inf((-(Z1-1)+sqrt(delta))/(2*Z2));

% check rmax<R 

if rmax>R
    warning('computation has to be rerun with higher R')
end

% r=linspace(0,1.3*rmax,100);
% plot(r,pol(r),'b',rmin,0,'ok',rmax,0,'or');

l1 = first_lyapunov(df, ddf, dddf, x, alpha, abs(eigenval_imag), rmin, bool_val);

st = dbstack();
if length(st)<2  % print only if called directly
    if all(abs(x)>10^-3) && abs(alpha)>10^-3
        fprintf('SUCCESS\n The Hopf bifurcation at [%1.3f,%1.3f], %1.3f has been verified\n',x(1),x(2),alpha)
    else
        fprintf('SUCCESS\n The Hopf bifurcation at [%1.3e,%1.3e], %1.3e has been verified\n',x(1),x(2),alpha)
    end
    fprintf('with a radius of %e to %e and Lyapunov coefficient %1.3f +/- %1.1e\n\n',rmin, rmax, mid(l1), rad(l1))
end

x_star = X(2+(1:N));
lambda_star = X(1);
eigenvec = eigenvec_real+1i*eigenvec_imag;
eigenval = 1i*eigenval_imag;
if l1>0
    stability = 1;
else
    stability = -1;
end

return


end

function l1 = first_lyapunov(df, ddf, dddf, x, alpha, eigenval_imag, rmin, bool_val)

% compute the first Lyapunov coefficient as in Kuznetsov
% p: DF(xH)^Tp=conj(beta)p
% p = null( DF(xH)^T-conj(beta)I)

df_mat=df(x,alpha);
df_mat_int=df(midrad(x,rmin),midrad(alpha,rmin));
% next line does not work
%p = null( df_mat.'-conj(1i*beta)*eye(size(df_mat)));

[all_q,all_beta]=eigs(df_mat, length(x));
all_eigs=diag(all_beta);
[min_diff,index_beta]=min(all_eigs-(1i*eigenval_imag));
if min_diff > rmin
    error('What is this case? Investigate')
end
q_new=all_q(:,index_beta);
% verification of the eigenvalue
[l,q]=verifyeig(df_mat_int,(1i*eigenval_imag),q_new);

if isnan(l)
    error('Linear validation failed')
end

[all_p,all_beta]=eigs(df_mat.', length(x));
all_eigs=diag(all_beta);
[~,index_beta]=min(all_eigs+(1i*eigenval_imag));
p=all_p(:,index_beta);
% verification of the eigenvalue
[l,p]=verifyeig(df_mat_int.',-(1i*eigenval_imag),p);

if isnan(l)
    error('Linear validation failed')
end

complex_product=@(a,b) sum(conj(a).*b);


% look for k, the rescaling coefficient such that <kp,q>=1
% k = k1 + i k2

% k1/k2
ratio = real(complex_product(p,q))/imag(complex_product(p,q));
k2 = 1/( ratio* real(complex_product(p,q))+imag(complex_product(p,q)));
k1= k2*ratio;

p=(k1+1i*k2)*p;

beta_ver=infsup(eigenval_imag-rmin,eigenval_imag+rmin);

% compute the first lyapunov coefficient
l1 = 1/(2*beta_ver.^2) * real(1i*complex_product(p,ddf(x,alpha,q)*q)*complex_product(p,ddf(x,alpha,q)*conj(q))+...
    beta_ver*complex_product(p,dddf(x,alpha,q,q)*conj(q)));

% OLD:
%l1 = 1/(2*beta_ver) * real(1i*complex_product(p,ddf(zeros(N,1),q,q)*complex_product(p,ddf(zeros(N,1),q,conj(q))))+...
%    beta_ver*complex_product(p,dddf(zeros(N,1),q,q,conj(q))));

% check the FLC is nonzero
if isnan(l1) && bool_val
    error('Could not validate Hopf, since the first Lyapunov coefficient is NaN')
elseif  intval(inf(l1))*intval(sup(l1))<0 && bool_val
    error('Could not validate Hopf, since the first Lyapunov coefficient is not verifyed to be non-zero')
end

% last check: all ther eigenvalues different then zero
[all_eigenvectors,all_eigenvalues]=eigs(df_mat.', length(x));
all_eigenvalues = diag(all_eigenvalues);
[~,index_beta1]=min(all_eigenvalues-(1i*eigenval_imag));
[~,index_beta2]=min(all_eigenvalues+(1i*eigenval_imag));
for i=1:length(all_eigs)
    if i~=index_beta1 && i~=index_beta2
        [L,~] = verifyeig(df_mat_int,all_eigenvalues(i),all_eigenvectors(:,i));
        if intval(inf(real(L)))*intval(sup(real(L)))<0 && bool_val
            error('Could not validate Hopf, since there is at least one eigenvalue that not verifyed to have non-zero real part')
        end
    end

end
end

