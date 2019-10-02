function [l, n] = linesearch_armijo(func, x, d)

epsilon = 0.3;  % 0 < epsilon < 1
alpha = 2;      % alpha > 1
lambda = 0.1;

% F(0) = func(x)
% F'(0) = d*func'(x)
h = 10^-10;

F = @(ll) func(x + ll.*d);
F0 = F(0);
Fderiv = (F(h)-F(0))/h;
T = @(lam) F0 + epsilon*lam*Fderiv;

%plot(lambda,T(lambda), 'b*');
%hold on;
fprintf('F(0) = %d \n', F0)


n = 0;
max_iterations = 100;
conditions = false;
    while ~conditions && n < max_iterations 
        conditions = true;
        
        if (T(lambda) < F(lambda))
           lambda = lambda/alpha;
           conditions = false;
           fprintf('F(lambda) = %d, T(lambda) =%d, n = %d \n',...
               F(lambda), T(lambda), n)

        end
        
        if (T(alpha*lambda) > F(alpha*lambda))
           lambda = lambda*alpha;
           conditions = false;
           fprintf('F(alpha*lambda) = %d, T(alpha*lambda) =%d, n = %d \n',...
               F(alpha*lambda), T(alpha*lambda), n)
        end
        %plot(lambda,T(lambda), 'r.');
        %hold on;
        fprintf('T(lambda) = %d, lambda = %d \n',T(lambda), lambda)
        


        n = n+1;
    end
l = lambda;
n

if n == max_iterations
    l = 0.1;
end

if isnan(F(lambda))
    error('Bad job of the line search, is NaN!')
end

if F(lambda)> F0
    error('Bad job of the line search, larger functional value!')
end

end
