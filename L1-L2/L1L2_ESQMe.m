function [x_new, iter] =  L1L2_ESQMe(A, b, delta, mu, M, xstart, d, alpha_init, L, maxiter, freq, tol)
%This code uses ESQM_e method solving the model
% min ||x||_1 - mu*||x||
% s.t. 1/2*||Ax - b||^2 - delta <=0  &&  \|x\|_inf <= M
%

% Input

% A               - m by n matrix (m << n)
% b                - m by 1 vector measurement
% delta           - real number > 0
% mu             - real number in (0, 1)
% M              - Upper bound of \|x\|
% xstart          - the starting point
% d                - real number > 0
% alpha_init   - real number > 0
% L                - the Lipschitz constant
% maxiter      - maximum number of iterations 
% freq            - The frequency of print the results
% tol              - tolerance


% Output

% x_new      - approximate stationary point
% iter           - number of iterations


% Initialization

theta = 1;   
theta0 = 1;
lambda = 0;  % Parameter for subproblem
iter = 0;
num_restart = 0;
re_freq = 200;
x_old = xstart;       % starting point x^{k}
x_new = x_old;    % starting point x^{x+1}


% fprintf(' ****************** Start   ESQMe ********************\n')
% fprintf('  iter        fval           gvalu      alpha       lambda      norm(x_new - x_old)    beta     norm(x_new)\n')


while  iter <= maxiter

    alpha = alpha_init;

    if norm(x_new) <= 1e-10
        xi = 0*x_new;
    else
        xi = mu*x_new/norm(x_new);
    end

    % iterations, gradient
    y0 = x_new + ((theta0 - 1)/theta) * (x_new - x_old);  
    y = y0 + (1/(L*alpha)).*xi;
    Ay0 = A * y0;
    tmpy0 = Ay0 - b;
    gvaly0 = (1/2)*norm(tmpy0)^2 - delta;
    grady0 = A'*tmpy0;
    x_old = x_new;

    % Solving the subproblem
    [x_new, lambda] = subprob_ESQM(y, grady0, grady0'*y0 - gvaly0, alpha, lambda, M, L); 

    %     if mod(iter, freq) == 0
    %         fprintf(' %5d| %16.10f|%3.3e|%3.3e|%3.3e|%3.3e|%3.3e|%3.3e\n',iter, norm(x_new,1)-norm(x_new), gvalx_new, alpha, lambda, norm(x_new - x_old), ((theta0 - 1)/theta),  norm(x_new) )
    %     end

    % check for termination

         if norm(x_new - x_old) < tol*max(norm(x_new),1)
            break
        end

    % Update theta
    theta0 = theta;
    theta = (1 + sqrt(1+4*theta0^2))/2;

    if re_freq < inf
        if (iter > 0 && (mod(iter, re_freq) == 0 || (y0 - x_new)'*(x_new - x_old) > 0) )
            num_restart = num_restart + 1;
            theta0 = 1;
            theta = 1;
        end
    end

    % Update alpha
    sss = gvaly0 + grady0'*(x_new - y0);
    if sss > 1e-10
        alpha_init = alpha + d;
    else
        alpha_init = alpha;
    end

    iter = iter + 1;

end



