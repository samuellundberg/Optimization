function [l, n] = linesearch(func, x, d)
%golden section linesearch that does not work too well

alpha = (sqrt(5) - 1) / 2;
n = 50;        %max amount of iterations
a = 0;  b = 1;      %0<lambda<1
lambda = a + (1-alpha)*(b - a);
mu     = a + alpha*(b - a);

F_l =@(ll) func(x + d*ll);
l = 0;

%for testing
lx = linspace(a, b);
% size(x)
% size(d*lx)
%plot(lx, F_l(lx))
% xlabel('lambda')
% ylabel('F(lambda)')
hold on

%make a termination criteria. l will be assigned exactly 1 time
for k = 1:n
    
    if sum(F_l(lambda).^2) > sum(F_l(mu).^2)
        a = lambda;
        lambda = mu;
        mu = a + alpha*(b - a);
    else
        b = mu;
        mu = lambda;
        lambda = a + (1-alpha)*(b - a);
    end
    if k == n
        if sum(F_l(lambda).^2) < sum(F_l(mu).^2)
            l = lambda;
        else
            l = mu;
        end
    end
end
% fels?ker 
% isnan(sum(F_l(lambda).^2))
% isnan(sum(F_l(mu).^2))
% sum(F_l(l).^2) > sum(F_l(0).^2)
% l
% d
% fel h?r
if isnan(sum(F_l(lambda).^2)) || isnan(sum(F_l(mu).^2)) || sum(F_l(l).^2) > sum(F_l(0).^2)
    if isnan(F_l(lambda))
        l = mu;
        fprintf('f lambda is nan')
    else
        l = lambda;
        fprintf('f mu is nan')
    end
    error('Bad job of the line search!')
end

%also for testing
x_new = linspace(a, b);
plot(x_new, F_l(x_new), '*')


end