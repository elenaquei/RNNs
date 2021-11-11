% numerical testing on Repeated maps with smooth step functions

K = [0, 0, 40];

% EXAMPLE 1

n_size = 2;
n_steps = 100;
size_sample = 1000;
scaling = 0.25;
for k = 1:K(1)
    W = 0.250*randn(n_size,n_size) - 1;
    b = 0.250*randn(n_size,1);
    
    x_stored = 50*randn(n_size,size_sample);
    
    y_stored = zeros(n_size, size_sample);
    for j = 1:size_sample
        y_stored(:,j) = repeated_map(x_stored(:,j), W, b, n_steps);
    end
    
    plot(y_stored(1,:),y_stored(2,:),'*')
    axis([-1,1,-1,1])
    pause(1)
end

% EXAMPLE 2

n_size = 2;
size_sample = 1000;
scaling = 0.25;

W = 3 * [0,1;-1,0] .* randn(2,2);
b = scaling * randn(n_size,1);

x_stored = 50 * randn(n_size,size_sample);
y_stored = x_stored;
plot(x_stored(1,:),x_stored(2,:),'*')
axis([-1,1,-1,1])
pause(1)
for k = 1:K(2)
    
    for j = 1:size_sample
        y_stored(:,j) = repeated_map(y_stored(:,j), W, b, 1);
    end
    
    plot(y_stored(1,:),y_stored(2,:),'*')
    axis([-1,1,-1,1])
    pause(1)
    if k > 2 && norm(y_old - y_stored)<10^-6
        break
    end
    y_old = y_stored;
end


% EXAMPLE 3

n_size = 2;
size_sample = 200;
scaling = 0.25;

% W = 0.3 * [1,1;0,10] .* randn(2,2); % interesting slow convergence - see
% the sigma function
% W = 0.3 * [1,5;0,10] .* randn(2,2);
W = 0.3 * [3,3;0,1] .* randn(2,2);
b = scaling * randn(n_size,1);

x_stored = 0.50 * randn(n_size,size_sample);
y_stored = x_stored;
plot(x_stored(1,:),x_stored(2,:),'*')
axis([-1,1,-1,1])
pause(1)
for k = 1:K(3)
    
    for j = 1:size_sample
        y_stored(:,j) = repeated_map(y_stored(:,j), W, b, 1);
    end
    
    plot(y_stored(1,:),y_stored(2,:),'*')
    axis([-1,1,-1,1])
    pause(1)
    if k > 2 && norm(y_old - y_stored)<10^-5
        break
    end
    y_old = y_stored;
end


return

function x = repeated_map(x, W, b, n)
sigma = @(x) tanh(x);
map = @(x, W, b) sigma(W * x + b);
for i = 1:n
    x = map(x, W, b);
end
end