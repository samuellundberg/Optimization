function gaussnewton(phi,t,y,start,tol,use_linesearch,printout,plotout)
% solves min(X) sum_{i=1}^m[phi(X,t_i) - y_i]^2 where
% phi(X,t) = x_1e^(-x_2*t) + x_3e^(-x_4*t) by the gauss newton method.
% in: phi: func.handle, t,y: given data, start: initial guess, tol:
% tolerence to terminate, the 3 last is 1/0 if you want it or not.
%example function call: gaussnewton(@phi2,t,y,[1;2;3;4],0.1,1,1,1);
% x is of dim n and t,y is of dim m
x = start;
r =@(x) phi(x, t) - y; %should really be a mx1 vector
% j_i,k = d(r_i)/d(x_k)
J =@(x) [exp(-x(2)*t), -t*x(1).*exp(-x(2)*t), exp(-x(4)*t), -t*x(3).*exp(-x(4)*t)];
% NG-step: x_k+1 = x_k - pinv(J(x_k))*r(x_k), r(x_k) = phi(x_k,T) - Y (mx1)
x_archive = [x];
c = 0;
while sum(r(x).^2) > tol        % what we want to minimize
    c = c + 1
    NG_dirr = (J(x)'*J(x)) \ (J(x)' * r(x));    %the last () to improve speed 
    
    if use_linesearch
        %linesearch i my old goldensection. allways make fix nbr of iter
        [lambda, ~] = linesearch_armijo(r, x, NG_dirr);
    else
        lambda = 1;
    end 
    x = x - lambda * NG_dirr;  
    x_archive = [x_archive x];
        
end
% How to build a string
if printout
    'The optimal solution is'
    sum(r(x).^2)
    'which we find in'
    x'
end
%some level curves would be nice
if plotout
    plot(x_archive(1, :), x_archive(2, :), '-')
    plot(x_archive(1, end), x_archive(2, end), '*')
end