function [amplitude, end_point] = detect_amplitude(f, eq, x0, T, epsilon)
% function amplitude = detect_amplitude(f, eq, x0, T, epsilon)
%
% given an ODE and its equilibrium this function computes an approximation
% of the amplitude of the STABLE periodic orbit around the equilibrium
% itself
%
% INPUT
% f         rhs of the ODE, autonomous
% eq        equilibrium
% x0        starting point
% T         transient time to reach the periodic orbit (DEFAULT = 10^3)
if nargin < 4
    T = 10^3;
end
if nargin < 5
    epsilon = 10^-1;
end
[~, y] = ode45( @(t,y)-f(y), [0,T], x0+epsilon);
[~, y] = ode45( @(t,y) f(y), [0,T], y(end,:));
sample_size = floor(size(y,1)/10);
amplitude = mean(sqrt(sum(abs(y(end-sample_size:end,:) - eq).^2,2)),1);
end_point = y(end,:);
end