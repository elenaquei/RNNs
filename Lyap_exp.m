% compute the maximal Lyapunov exponent for the lorenz attractor

% the lorenz attractor
sigma = 10;
beta = 8/3;
rho = 28;

f = @(t,x)[ sigma *( x(2) - x(1) )
    x(1)*( rho - x(3) ) - x(2)
    x(1)*x(2) - beta * x(3)];


% computing the Lyapunov exponent
N = 100;
k = 5;
T = 30;
epsilon = 10^-4;
dim_neighborhood = (2*epsilon)^3;
lyapExp = 0;
for j = 1:N
    x_n0 = randn(3,1) * 20;
    x_0 = x_n0-epsilon + epsilon *rand(3,k);
    [t,y] = ode45(f, [0,T],x_n0);
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

