function [x, iter] =  L1L2_ESQM_ls_Lor(A, b, sigma, mu, M, xstart, d, alpha_init, L, gamma, freq, tol, maxiter)
% This aims to use ESQM to find the minimizer of the following problem (involve Lorentzion norm)
% min ||x||_1 - mu*||x||
% s.t.\|Ax - b\|_{LL_2,gamma} <= sigma  &&  \|x\|_inf <= M
% It uses ESQM
%
% Input
%
% A                 - m by n matrix (m << n)
% b                   - m by 1 vector measurement
% sigma             - real number > 0
% mu                 - real number in (0, 1)
% M                  - Upper bound of \|x\|
% xstart              - the starting point
% d                    - real number > 0
% alpha_init        - real number > 0
% L                    - the Lipschitz constant
% gamma           - real number > 0
% freq               - The frequency of print the results
% tol                  - tolerance 
% maxiter           - maximum number of iterations [inf]
%
%
% Output
%
% x           - approximate stationary point
% iter        - number of iterations


% Initialization

rho = 1e-4;
lambda = 0; % parameter for subproblem
iter = 0;

% Compute function value and gradient at start point
x = xstart;
Ax = A*x;
tmp = Ax - b;
ellx = sum(log(1 + tmp.^2/gamma^2)) - sigma;
wx = 1./(gamma^2 + tmp.^2);
grad = 2*(A'*(tmp.*wx)); % gradient of \ell
newsigma1 = grad'*x - ellx; % value of subproblem's inequlity constraint
fval = norm(x, 1) - mu*norm(x);


% fprintf(' ****************** Start   ESQMe ********************\n')
%fprintf('  iter        fval      gvalu      alpha       lambda      norm(x_new - x_old)      beta      norm(x_new)\n')


while   iter < maxiter

    alpha = alpha_init;

    if norm(x) <= tol
        xi = 0*x;
    else
        xi = mu*x/norm(x);
    end

    % iterations, gradient

    y = x + (1/(L*alpha)).*xi;

    [u, lambda] = subprob_L1L2_ESQMe_Lor(y, grad, newsigma1, alpha, lambda, M, L); % Solving the subproblem


    %  Line search
    Au = A*u;
    fval_new = fval + alpha*max(0, ellx);
    iter1 = 0;
    t = 1;
    while 1 == 1
        xtest = x + t*(u - x);
        Axtest = Ax + t*(Au - Ax);
        tmpxtest = Axtest - b;
        fvalxtest = norm(xtest, 1) - mu*norm(xtest);
        ellxtest = sum(log(1 + tmpxtest.^2/gamma^2)) - sigma;
        fvalxtest_new = fvalxtest + alpha*max(0, ellxtest);

        if fvalxtest_new - fval_new > -alpha*rho*t*norm(u - x)^2 && t > 1e-10
            t = t/2;
            iter1 = iter1 + 1;
        else
            break
        end
    end

    %    if mod(iter,freq) == 0
    %         fprintf(' %5d|%5d| %16.10f|%3.4e|%3.4e|%3.4e|%3.4e|%3.4e|%3.4e\n',iter, iter1, fval, err1, err2, norm( u - x), beta,norm(grad), t )
    %     end


    % check for termination
    if norm(u - x) <= tol*max(1, norm(u)) || t <= 1e-10
        x = u;
        break
    end



    % Update alpha
    sss = ellx + grad'*(u - x);
    if sss > 1e-10
        alpha_init = alpha + d;
    else
        alpha_init = alpha;
    end

    % Update iterations, gradient and function value

    x = xtest;
    Ax = Axtest;
    tmp = tmpxtest;
    ellx = ellxtest;
    wx = 1./(gamma^2 + tmp.^2);
    grad = 2*(A'*(tmp.*wx));
    newsigma1 = grad'*x - ellx;
    fval = fvalxtest;


    iter = iter + 1;

end



