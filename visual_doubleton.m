% visualization of doubletons

dim = 2;

x = intval(rand(dim,1));

a = intervals([3;-4]-0.1, [3;-4]+0.1);

C = intervals([1,2;2,3]-0.1, [1,2;2,3]+0.1);

% visualization of x + C * a
figure;
plot_point(x + C * a, 'r');
hold on
plot_mat_mult(mid(x), C, a)
hold on



% ====== end of file ======

function C = intervals(A, B)
min_mat = min(A,B);
max_mat = max(A,B);
C = infsup(min_mat, max_mat);
end

function plot_point(x,c)
fill( [inf(x(1)), inf(x(1)), sup(x(1)), sup(x(1))], ...
    [inf(x(2)) sup(x(2)) sup(x(2)) inf(x(2))],c)
hold on
plot(mid(x(1)), mid(x(2)), '*')
end

function plot_mat_mult(x, C, a)
easy_fill(x+inf(C)*inf(a),x+sup(C)*inf(a),x+inf(C)*sup(a),x+sup(C)*sup(a),'b')
hold on
easy_fill(x+inf(C)*inf(a),x+inf(C)*sup(a),x+sup(C)*inf(a),x+sup(C)*sup(a),'b')
end

function easy_fill(a,b,c,d, color)
fill([a(1), b(1), c(1), d(1)],[a(2), b(2), c(2), d(2)], color) 

end