
%first test of linesearch
a = -2; % try a = -2 too, then a = 5 and -5, then a = 10 and -10
func = @(x)(1-10.^(a*x)).^2;
func = testf

x = 1;
d =  -1; %no point in making a direction when it it 1d
[l, n] = linesearch_armijo(func, x, d);
x+l*d
func(x + l*d)

%%


% Test 1: use 
x=[0;0]; d=[1;0]; 
% Test 2: use x=[0;0], d=[0;1]
f =@(x) (1e58*x(1)-1)^100+(1e-58*x(2)-1)^2-2;
[l, n] = linesearch(f, x, d);

%% 
[y, t] = data1;
%x=[1 1 1 1]';
%y = phi2(x,t);
gaussnewton(@phi2,t,y,[1;2;3;4],0.1,1,1,1);

%% trying on phi2
[y, t] = data1;
x = [1;2;3;4];
d = [86.3514  60.7932 -71.6341 59.2824]';
lam = linspace(0, 1);
r =@(x) sum((phi2(x, t) - y).^2)
r(x + d*lam(1))
hold on
v = [];
for i = 1:length(lam)
    v= [v r(x + d*lam(i))];
end
plot(lam,v)