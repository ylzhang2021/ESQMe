function [xstar, lambda] = subprob_ESQM(y, a, sigma, alpha, lambda, M, L)

% This aims to find the minimizer of the following problem
% min   \|x\|_1 + alpha*t + alpha*L/2\|x - y\|^2
% s.t. <a, x> - sigma <= t && \|x\|_inf <= M && t >= 0

% Input

% y              - n by 1 vector measurement
% a              - n by 1 vector measurement
% sigma       - real number > 0
% alpha        - real number > 0
% lambda     - real number which is the initial of lambda
% M             - real number > 0
% L              - the Lipschitz constant


% Output

% xstar       - approximate stationary point
% lambda   - corresponding Lagrange multipliers

tol = 1e-8;


%Calculate x^* when lambda = 0
x_newstar = sign(y).*min(max(abs(y) - 1/(alpha*L), 0), M);   
g0 = sigma - sum(a.*x_newstar);  

%Calculate x^* when  lambda = alpha
tmp1 = y - (1/L).*a;
x_newstar1 = sign(tmp1).*min(max(abs(tmp1) - 1/(alpha*L), 0), M); 
g1 = sigma - sum(a.*x_newstar1);

if g0 > -tol % Check lambda = 0
    lambda = 0;
    xstar = x_newstar;  
    return
elseif g1 < tol % Check  lambda = alpha
    lambda = alpha;
    xstar = x_newstar1;  
    return
else
    [xstar, lambda] = Newton_Monotone(y, a, sigma, alpha*L, lambda, M, tol);
end

end