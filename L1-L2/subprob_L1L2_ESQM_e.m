function [xstar, lambda] = subprob_L1L2_ESQM_e(y, a, sigma, alpha, lambda, M, L)

% This aims to find the minimizer of the following problem
% min   \|x\|_1 - mu*<xi, x> + alpha*t + alpha*L/2\|x - y\|^2
% s.t. <a, x> - sigma <= t && \|x\|_inf <= M && t >= 0

% Input
%
% x0            - n by 1 vector measurement
% a              - n by 1 vector measurement
% sigma      - real number > 0
% mu         - real number  [ 0.1 ]
% alpha     - real number > 0
% lambda   - real number which is the initial of lambda
% J              - a positive integer whic denote the size of each block
% M           - real number > 0
%
%
% Output
%
% xstar       - approximate stationary point
% lambda   - corresponding Lagrange multipliers

tol = 1e-10;


%Calculate x^* when lambda = 0?
x_newstar = sign(y).*min(max(abs(y)-(1/(alpha*L)),0),M);   %¼ÆËãx^{k+1}
g0 = sigma - sum(a.*x_newstar); % T(lambda)

%Calculate x^* when  lambda = alpha?
tmp1 = y - (1/L).*a;
x_newstar1 = sign(tmp1).*min(max(abs(tmp1)-(1/(alpha*L)),0),M); %¼ÆËãx^{k+1}
g1 = sigma - sum(a.*x_newstar1);

if g0 > -tol % Check lambda = 0?
    lambda = 0;
    xstar = x_newstar;  % x_new
    return
elseif g1 < tol % Check  lambda = alpha?
    lambda = alpha;
    xstar = x_newstar1;  % x_new
    return
else
    [xstar, lambda] = Newton_Monotone_GL_e(y, a, sigma, alpha, lambda, M, tol, L);
end

end