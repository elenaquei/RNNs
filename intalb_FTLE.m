% compute the finite time Lyapunov exponent of the logistic map

x0 = affari(rand());

N = 30;
r = 3.8;
f = @(x) r * x * (1-x); 
fn_x = f(x0);

for i = 1:N
    fn_x = f(fn_x);
    disp(fn_x)
end

