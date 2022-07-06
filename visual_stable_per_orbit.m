% display stable periodic orbit found in 4D system with tanh

load('stable_periodic_orbit.mat')
% this loads R1 and R2 that sustain a periodic orbit
dim = size(R1, 1); 
if dim ~= 4
    disp('Something changed in loaded file since last time, the code will still try to run');
    disp('Figures might be less informative, because the plots have been fine tuned to other parameters.');
end
[R1, R2,solutions, positive_lyap_index, negative_lyap_index, ...
    positive_lyap, negative_lyap, unproven] = asym_RHS_Hopf...
    (dim, 'R1', R1, 'R2', R2);

if isempty(negative_lyap)
    disp('No stable periodic orbit has been found, the program quits')
    return
end

index = negative_lyap_index(1);

W = @(a) R1 - R1.' + a * eye(dim) + R2;
f = @(x, a) asym_rhs(x, W(a));

% visualization : find last Hopf bifurcation
x = solutions(1:end,2+(1:dim));
bifurcation_values = solutions(1:end,1);

alpha_bif = bifurcation_values(index);

eigenvec  = solutions(1:end,2+dim+(1:dim))+ 1i*solutions(1:end,2+2*dim+(1:dim));
plotting_dim = min(dim, 6);

min_exp= 8;
end_times = [800, 1500, 10*ones(1, min_exp-2)];
for i = min_exp:-1:1
    y = x(index,:) + 10^-14 * abs(eigenvec(index,:));
    figure
    for j = 1:10
        [t,y] = ode45(@(t,x) f(x, bifurcation_values(index) + 10^(-i)), [0,end_times(i)], y(end,:));
    end
    plot(t, y(:,1:plotting_dim),'LineWidth',3)
    plot(t(1:end), y(1:end,1:plotting_dim),'LineWidth',3)
    title(sprintf('%f',10^(-i)))
end


% helper functions 

function h_dot = asym_rhs(h, W)
h_dot = tanh( W * h );
end

function h_dot = asym_rhs_sin(h, W)
% sine activation function for testing purposes
h_dot = sin( W * h );
end