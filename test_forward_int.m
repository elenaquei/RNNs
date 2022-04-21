% rigorous forward integration of non-autonomous systems


%% autonomous example
% 1D - more D would requirebeing carful with vector components

% normal derivatives (case specific)
% f = @(y) -sin(y);
% f1 = @(y) -cos(y);
% f2 = @(y) sin(y);
% f3 = @(y) cos(y);
f = @(y) y;
f1 = @(y) 1;
f2 = @(y) 0;
f3 = @(y) 0;

% total derivatives - general 
f1_full = @(y) f1(y) * f(y);
f2_full = @(y) f2(y) * f(y)^2 + (f1(y))^2 * f(y);
f3_full = @(y) f3(y) * f(y)^3 + 2* f1(y) * f2(y) * f(y)^2 + (f1(y))^3 * f(y);


y_0 = 1;
y0 = intval(y_0);
t0 = 0;
h = 0.01;
tend = 10;

iter = floor((tend-t0)/h);

y_stored = intval(zeros(1,iter));
t_stored = intval(zeros(1,iter));
for i = 1:iter-1
    y_stored(i) = y0;
    t_stored(i) = t0;
    [y, R] = autonomousEuler(y0, h, f, f1_full, f2_full, f3_full); 
    % order 4 approximation 
    y0 = infsup(inf(y)-sup(R), sup(y)+sup(R));
    t0 = t0 + h;
end

y_stored(i+1) = y0;
t_stored(i+1) = t0;

% compare with ODE45
f_ode45 = @(y,t) f(y);
[tsol, ysol] = ode45(f_ode45, mid(t_stored), y_0);


plot(mid(t_stored), sup(y_stored),'+')
hold on
plot(mid(t_stored), inf(y_stored),'-')


plot(tsol, ysol, 'g')
plot(t_stored, exp(t_stored), 'r*')
legend('uper bound', 'lower bound', 'ode45', 'exact exp')

return




%% non-autonomous example 
f = @(y,g) g*y;
g  = @(t) sin(t);
gprime = @(t) cos(t);
d1f = @(y, g) y;
d2f = @(y,g) g;

y0 = intval(1);
t0 = 0;
h = 0.0001;

iter = 100;
y_stored = intval(zeros(1,iter));
t_stored = intval(zeros(1,iter));
for i = 1:iter-1
    y_stored(i) = y0;
    t_stored(i) = t0;
    [y, R] = Euler(y0, t0, f, d1f, d2f, g, gprime, h);
    y0 = infsup(inf(y)-sup(R), sup(y)+sup(R));
    t0 = t0 + h;
end

y_stored(i+1) = y0;
t_stored(i+1) = t0;

plot(mid(t_stored), sup(y_stored),'+')
hold on
plot(mid(t_stored), inf(y_stored),'-')
    
function yInt = vecInt(y0, y1)
if isa(y0,'intval')
    y0_inf = inf(y0);
    y0_sup = sup(y0);
else
    y0_inf = y0;
    y0_sup = y0;
end
if isa(y1,'intval')
    y1_inf = inf(y1);
    y1_sup = sup(y1);
else
    y1_inf = y1;
    y1_sup = y1;
end
yInt = infsup(min(y0_inf, y1_inf), max(y0_sup,y1_sup));
end

function [y, R] = Euler(y0, t0, f, d1f, d2f, g, gprime, h)

y = y0 + h * f(y0, g(t0));

tInt = infsup(t0, t0+h);
yInt = vecInt(y0, y);
gInt = g(tInt);

R = 1/2*( d1f(yInt, gInt) * f(yInt, gInt) + d2f(yInt, gInt) * gprime(tInt));

end


function [y,R] = autonomousEuler(y0, h, f, f1, f2, f3)
y = y0 + h*f(y0) + 1/2 * h^2 * f1(y0) + 1/6 * h^3 *f2(y0);

yInt = vecInt(y, y0);
R = 1/(2*3*4)*h^4*abs(f3(yInt));
end
