function [x_new, iter] =  L1L2_ESQM_e(A, b, delta, mu, M, xstart, d, alpha_init, L, maxiter, freq, tol)
%This code uses ESQM_e method solving the model
% min ||x||_1 - mu*||x||
% s.t. 1/2*||Ax - b||^2 - delta <=0  &&  \|x\|_inf <= M
%
% Input
%
% A            - m by n matrix (m << n)
% b             - m by 1 vector measurement
% sigma      - real number > 0
% mu          - real number in (0, 1)
% J              - a positive integer whic denote the size of each block
% xstart       - the starting point
% delta       - real number > 0
% L             - the Lipschitz constant
% M           - Upper bound of \|x_J\| for any J
% maxiter   - maximum number of iterations [inf]
% freq         - The frequency of print the results
% tol           - tolerance [1e-8]
%
%
% Output
%
% x            - approximate stationary point
% iter        - number of iterations


% Initialization
% alpha_init =1; % alpha

theta = 1;  %…Ë÷√beta
theta0 = 1;
lambda = 0; % parameter for subproblem
iter = 0;
num_restart = 0;
re_freq = 200;
% n = size(xstart, 1);
x_old = xstart; % starting point x^{k}
x_new = x_old;% starting point x^{x+1}



% fprintf(' ****************** Start   ESQMe ********************\n')
% fprintf('  iter        fval        err1      err2      gvalu      alpha       lambda      norm(x_new - x_old)      beta      norm(x_new)\n')


while  iter <= maxiter
    
    alpha = alpha_init; 
    
    if norm(x_new) <= tol
    xi = 0*x_new;
    else
    xi = mu*x_new/norm(x_new);
    end
    
    % iterations, gradient 
    y0 = x_new + ((theta0 - 1)/theta) * (x_new - x_old);  %º∆À„y^k
    y = y0 + (1/(L*alpha)).*xi;
    Ay0 = A * y0;
    tmpy0 = Ay0 - b;
    gvaly0 = (1/2)*norm(tmpy0)^2 - delta;
    grady0 = A'*tmpy0;

    
    x_old = x_new;
    [x_new, lambda] = subprob_L1L2_ESQM_e(y, grady0, grady0'*y0 - gvaly0, alpha, lambda, M, L); % Solving the subproblem
%     Ax_new = A*x_new; % A*x^{k+1}
%     tmpx_new = Ax_new - b; 
%     gvalx_new = (1/2)*norm(tmpx_new)^2 - delta;
%     fval = norm(x_new, 1) - mu*norm(x_new);
    
    

    
%     if mod(iter, freq) == 0
%         fprintf(' %5d| %16.10f|%3.3e|%3.3e|%3.3e|%3.3e|%3.3e|%3.3e\n',iter, fval, gvalx_new, alpha, lambda, norm(x_new - x_old), ((theta0 - 1)/theta),  norm(x_new) )
%     end
    
     % check for termination
     if norm(x_new - x_old) < tol*max(norm(x_new),1)
        break
    end

      % Update theta
    theta0 = theta;
    theta = (1 + sqrt(1+4*theta0^2))/2;

    if re_freq < inf
        if (iter > 0 && (mod(iter, re_freq) == 0 || (y0-x_new)'*(x_new-x_old) > 0) )
            num_restart = num_restart + 1;
            theta0 = 1;
            theta = 1;
        end
    end
  
    sss = gvaly0 + grady0'*(x_new - y0); 
    if sss > 1e-10
        alpha_init = alpha + d;
    else
        alpha_init = alpha;
    end
    
    iter = iter + 1;
    
end



