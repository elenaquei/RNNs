% compute the maximal Lyapunov exponent for the lorenz attractor
% doesn't work

% the lorenz attractor
sigma = 10;
beta = 8/3;
rho = 28;

f = @(t,x)[ sigma *( x(2) - x(1) )
    x(1)*( rho - x(3) ) - x(2)
    x(1)*x(2) - beta * x(3)];


% computing the Lyapunov exponent
N = 1;
k = 40;
T = 40;
epsilon = 10^-4;
dim_neighborhood = (2*epsilon)^3;
lyapExp = 0;
x_n0 = randn(3,N) * 20;
for j = 1:N
    x_0 = x_n0(:,j) - epsilon/2 + epsilon *rand(3,k);
    [t,y] = ode45(f, [0,T],x_n0(:,j));
    x_deltaN = y(end,:);
    y_deltaN = 0*x_0;
    lyapExp_sumsum = 0;
    for jj = 1:k
        [t,y] = ode45(f, [0,T],x_0(:,jj));
        y_deltaN(:,jj) = y(end,:);
        plot3(y(:,1),y(:,2),y(:,3))
        hold on
        plot3(y(end,1),y(end,2),y(end,3),'*')
        plot3(y(1,1),y(1,2),y(1,3),'d')
        lyapExp_sumsum = lyapExp_sumsum + norm(y_deltaN(:,jj) - x_deltaN);
    end
    lyapExp = lyapExp + 1/N * log(lyapExp_sumsum/dim_neighborhood);
end
disp(lyapExp)

