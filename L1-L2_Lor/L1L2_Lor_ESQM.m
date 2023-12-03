function [x_new, iter] =  L1L2_Lor_ESQM(A, b, sigma, mu, M, xstart, d, alpha_init, L, gamma, freq, tol, maxiter)
% This aims to use ESQM to find the minimizer of the following problem (involve Lorentzion norm)
% min ||x||_1 - mu*||x||
% s.t.\|Ax - b\|_{LL_2,gamma} <= sigma  &&  \|x\|_inf <= M

% Input

% A                - m by n matrix (m << n)
% b                 - m by 1 vector measurement
% sigma           - real number > 0
% mu              - real number in (0, 1)
% M               - Upper bound of \|x\|
% xstart          - the starting point
% d                - real number > 0
% alpha_init    - real number > 0
% L                - the Lipschitz constant
% gamma         - real number > 0
% freq            - The frequency of print the results
% tol            - tolerance
% maxiter     - maximum number of iterations [inf]

% Output

% x_new           - approximate stationary point
% iter        - number of iterations

% Initialization

lambda = 0;
iter = 0;
x_old = xstart;
x_new = x_old;

% fprintf(' ****************** Start   ESQMe ********************\n')
%fprintf('  iter        fval      gvalu      alpha       lambda      norm(x_new - x_old)       norm(x_new)\n')


while   1 == 1

    alpha = alpha_init;

    if norm(x_old) <= 1e-10
        xi = 0*x_old;
    else
        xi = mu*x_old/norm(x_old);
    end

    % iterations, gradient

    Ax_new = A*x_new;
    tmpx_new = Ax_new - b;
    gvalx_new = sum(log(1 + tmpx_new.^2/gamma^2)) - sigma;
    wx_new = 1./(gamma^2 + tmpx_new.^2);
    gradx_new = 2*(A'*(tmpx_new.*wx_new));   % gradient of g
    newsigma1 =  gradx_new'*x_new - gvalx_new;

    y = x_new + (1/(L*alpha)).*xi;
    x_old = x_new;

    % Solving the subproblem
    [x_new, lambda] = subprob_ESQM(y, gradx_new, newsigma1, alpha, lambda, M, L);


    %     if mod(iter, freq) == 0
    %         fprintf(' %5d| %16.10f|%3.3e|%3.3e|%3.3e|%3.3e|%3.3e\n',iter, fval, gvalx_new, alpha, lambda, norm(x_new - x_old),  norm(x_new) )
    %     end

    % check for termination
    if norm(x_new - x_old) < tol*max(norm(x_new), 1) || iter >= maxiter
        break
    end


    % Update alpha
    sss = gvalx_new + gradx_new'*(x_new - x_old);
    if sss > 1e-10
        alpha_init = alpha + d;
    else
        alpha_init = alpha;
    end

    iter = iter + 1;

end



