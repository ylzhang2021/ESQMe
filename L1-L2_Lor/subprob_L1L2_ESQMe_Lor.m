function [xstar, lambda] = subprob_L1L2_ESQMe_Lor(y, a, sigma, alpha, lambda, M, L)

% This aims to find the minimizer of the following problem
% min   \|x\|_1 + alpha*t + alphaL/2\|x - y\|^2
% s.t. <a, x> - sigma <= t && \|x\|_inf <= M && t >= 0

% Input
%
% y             - n by 1 vector measurement
% a              - n by 1 vector measurement
% sigma      - real number > 0
% alpha     - real number > 0
% lambda   - real number which is the initial of lambda
% M           - real number > 0
% L             - a constant [ \|A\|^2 ]
%
%
% Output
%
% xstar       - approximate stationary point
% lambda   - corresponding Lagrange multipliers

tol = 1e-10;


%Calculate x^* when lambda = 0?
xstar0 = sign(y).*min(max(abs(y) - (1/(L*alpha)), 0), M);
g0 = sigma - a'*xstar0;

%Calculate x^* when lambda = alpha
tmpalpha = y - (1/L).* a;
xstaralpha = sign(tmpalpha).*min(max(abs(tmpalpha) - (1/(L*alpha)), 0), M);
gbeta = sigma - a'*xstaralpha;


if g0 > -tol % Check lambda = 0?
    lambda = 0;
    xstar = xstar0;
    return
elseif gbeta < tol % Check lambda = alpha?
    lambda = alpha;
    xstar = xstaralpha;
    return
else
    [xstar, lambda] = Newton_Monotone_LL(y, a, sigma, alpha, lambda, M, tol, L);
end

end