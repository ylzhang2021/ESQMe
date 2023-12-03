function [x, iter] =  L1L2_Lor_ESQM_ls(A, b, sigma, mu, M, xstart, d, alpha_init, gamma, freq, tol, maxiter)
% This aims to use ESQM_ls to find the minimizer of the following problem (involve Lorentzion norm)
% min ||x||_1 - mu*||x||
% s.t.\|Ax - b\|_{LL_2,gamma} <= sigma  &&  \|x\|_inf <= M


% Input

% A                 - m by n matrix (m << n)
% b                   - m by 1 vector measurement
% sigma             - real number > 0
% mu                 - real number in (0, 1)
% M                  - Upper bound of \|x\|
% xstart              - the starting point
% d                    - real number > 0
% alpha_init        - real number > 0
% gamma           - real number > 0
% freq               - The frequency of print the results
% tol                  - tolerance
% maxiter           - maximum number of iterations [inf]


% Output

% x           - approximate stationary point
% iter        - number of iterations


% Initialization

rho = 1e-4;
lambda = 0; % parameter for subproblem
iter = 0;

% Compute function value and gradient at start point
x = xstart;
Ax = A*x;
tmpx = Ax - b;
gvalx = sum(log(1 + tmpx.^2/gamma^2)) - sigma;
wx = 1./(gamma^2 + tmpx.^2);
gradx = 2*(A'*(tmpx.*wx));        % gradient of g
newsigma1 = gradx'*x - gvalx;
fval = norm(x, 1) - mu*norm(x);


% fprintf(' ****************** Start   ESQMe ********************\n')
%fprintf('  iter        fval      gvalu      alpha       lambda      norm(x_new - x_old)         norm(x_new)\n')


while   iter < maxiter

    alpha = alpha_init;

    if norm(x) <= 1e-10
        xi = 0*x;
    else
        xi = mu*x/norm(x);
    end

    y = x + 1/alpha.*xi;

    % Solving the subproblem
    [u, lambda] = subprob_ESQM(y, gradx, newsigma1, alpha, lambda, M, 1);


    %  Line search
    Au = A*u;
    fval_new = fval + alpha*max(0, gvalx);
    iter1 = 0;
    t = 1;
    while 1 == 1
        xtest = x + t*(u - x);
        Axtest = Ax + t*(Au - Ax);
        tmpxtest = Axtest - b;
        fvalxtest = norm(xtest, 1) - mu*norm(xtest);
        gvalxtest = sum(log(1 + tmpxtest.^2/gamma^2)) - sigma;
        fvalxtest_new = fvalxtest + alpha*max(0, gvalxtest);

        if fvalxtest_new - fval_new > -alpha*rho*t*norm(u - x)^2 && t > 1e-8
            t = t/2;
            iter1 = iter1 + 1;
        else
            break
        end
    end

%        if mod(iter,freq) == 0
%             fprintf(' %5d|%5d| %16.10f|%3.4e|%3.4e|%3.4e|%3.4e\n',iter, iter1, fval, gvalx, norm( u - x), norm(gradx), t)
%         end


    % check for termination
    if norm(xtest - x) < tol*max(1, norm(xtest)) || t <= 1e-8
        if t  <= 1e-8
            fprintf(' Terminate due to small t\n')
        end
        break
    end

    % Update alpha
    sss = gvalx + gradx'*(u - x);
    if sss > 1e-10
        alpha_init = alpha + d;
    else
        alpha_init = alpha;
    end

    % Update iterations, gradient and function value

    x = xtest;
    Ax = Axtest;
    tmpx = tmpxtest;
    gvalx = gvalxtest;
    wx = 1./(gamma^2 + tmpx.^2);
    gradx = 2*(A'*(tmpx.*wx));
    newsigma1 = gradx'*x - gvalx;
    fval = fvalxtest;

    iter = iter + 1;

end



