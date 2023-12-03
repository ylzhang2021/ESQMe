function [x, iter] =  L1L2_ESQM_ls(A, b, delta, mu, M, xstart, d, alpha_init, maxiter, freq, tol)
%This code uses ESQM method solving the model
% min ||x||_1 - mu*||x||
% s.t. 1/2*||Ax - b||^2 - delta <=0  &&  \|x\|_inf <= M

% Input

% A               - m by n matrix (m << n)
% b                - m by 1 vector measurement
% delta           - real number > 0
% mu             - real number in (0, 1)
% M              - Upper bound of \|x\|
% xstart          - the starting point
% d                - real number > 0
% alpha_init   - real number > 0
% maxiter      - maximum number of iterations 
% freq            - The frequency of print the results
% tol              - tolerance 
%
%
% Output
%
% x              - approximate stationary point
% iter           - number of iterations


% Initialization

rho = 1e-4;
lambda = 0; % Parameter for subproblem
iter = 0;
x = xstart; 

%  gradient of the starting point
Ax = A*x;
tmpx = Ax - b;
gradx = A'*tmpx;
gvalx = (1/2)*norm(tmpx)^2 - delta;
fval = norm(x, 1) - mu*norm(x);


% fprintf(' ****************** Start   ESQMe ********************\n')
% fprintf('  iter        fval           gvalu      alpha       lambda      norm(x_new - x_old)          norm(x_new)\n')


while  iter <= maxiter

    alpha = alpha_init;

    if norm(x) <= 1e-10
        xi = 0*x;
    else
        xi = mu*x/norm(x);
    end

    y = x + 1/alpha.*xi;

    %Solving the subproblem
    [u, lambda] = subprob_ESQM(y, gradx, gradx'*x - gvalx, alpha, lambda, M, 1); 

    % Line search

    Au = A*u;
    fval_new = fval + alpha*max(0, gvalx);
    iter1 = 0;
    t = 1;
    while 1== 1
        xtest = x + t*(u - x);
        Axtest = Ax + t*(Au - Ax);
        tmpxtest = Axtest - b;
        gvalxtest = (1/2)*norm(tmpxtest)^2 - delta;
        fvalxtest= norm(xtest, 1) -  mu*norm(xtest);
        fvaltest_new = fvalxtest + alpha*max(0, gvalxtest);

        if fvaltest_new - fval_new  > - alpha*rho*t*norm(u - x)^2 && t>1e-8
            t = t/2;
            iter1 = iter1 +1;
        else
            break
        end
    end
   

        
    %     if mod(iter, freq) == 0
    %         fprintf(' %5d| %16.10f|%3.3e|%3.3e|%3.3e|%3.3e|%3.3e|%3.3e\n',iter, fval, gvalx, alpha, lambda, norm(u - x),  norm(u) )
    %     end
    %

    % check for termination

    if  norm(xtest - x) < tol*max(norm(xtest), 1)  || t<=1e-8
        if t < 1e-8
            fprintf(' Terminate due to small gamma\n')
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
    gradx = A'*tmpx;
    gvalx = gvalxtest;
    fval = fvalxtest;

    iter = iter + 1;

end



