function [x_new, iter] =  L1L2_ESQMe_Lor(A, b, sigma, mu, M, xstart, d, alpha_init, L, gamma, freq, tol)
% This aims to use ESQMe to find the minimizer of the following problem (involve Lorentzion norm)
% min ||x||_1 - mu*||x||
% s.t.\|Ax - b\|_{LL_2,gamma} <= sigma  &&  \|x\|_inf <= M
% It uses ESQM_e
%
% Input
%
% A           - m by n matrix (m << n)
% b           - m by 1 vector measurement
% sigma       - real number > 0
% mu          - real number in (0, 1)
% xstart      - the starting point
% d           - real number > 0
% L           - the Lipschitz constant
% M           - Upper bound of \|x\|
% maxiter     - maximum number of iterations [inf]
% freq        - The frequency of print the results
% tol         - tolerance [1e-4]
%
%
% Output
%
% x           - approximate stationary point
% iter        - number of iterations


% Initialization
% alpha_init = 0.05; % alpha

theta = 1;  %…Ë÷√beta
theta0 = 1;
lambda = 0; % parameter for subproblem
iter = 0;
num_restart = 0;
re_freq = 49;
x_old = xstart; % starting point x^{k}
x_new = x_old;% starting point x^{x+1}

[~,n] = size(A);




% fprintf(' ****************** Start   ESQMe ********************\n')
%fprintf('  iter        fval      gvalu      alpha       lambda      norm(x_new - x_old)      beta      norm(x_new)\n')


while   1 == 1
    
    alpha = alpha_init; 
    
    if norm(x_new) <= tol
    xi = 0*x_new;
    else
    xi = mu*x_new/norm(x_new);
    end
    
    % iterations, gradient 
    y0 = x_new + ((theta0 - 1)/theta) .* (x_new - x_old);  %º∆À„y^k
    Ay0 = A * y0;
    tmpy0 = Ay0 - b;
    elly0 = sum(log(1 + tmpy0.^2/gamma^2)) - sigma; 
    wy0 = 1./(gamma^2 + tmpy0.^2);
    grady0 = 2*(A'*(tmpy0.*wy0)); % gradient of \ell
    newsigma1 =  grady0'*y0 - elly0;
     y = y0 + (1/(L*alpha)).*xi;
    
    x_old = x_new; 
    
    [x_new, lambda] = subprob_L1L2_ESQMe_Lor(y, grady0, newsigma1, alpha, lambda, M, L); % Solving the subproblem
    

%     fval = norm(x_new, 1) - mu*norm(x_new);
%     if mod(iter, freq) == 0
%         fprintf(' %5d| %16.10f|%3.3e|%3.3e|%3.3e|%3.3e|%3.3e|%3.3e\n',iter, fval, elly0, alpha, lambda, norm(x_new - x_old), (theta/theta0 - 1/theta),  norm(x_new) )
%     end
    
     % check for termination
     if norm(x_new - x_old) < tol*max(norm(x_new), 1)
        break
    end

      % Update theta
    theta0 = theta;
    theta = (1 + sqrt(1+4*theta^2))/2;

    if re_freq < inf
        if (iter > 0 && (mod(iter, re_freq) == 0 || (y0 - x_new)'*(x_new - x_old) > 0) )
            num_restart = num_restart + 1;
            theta0 = 1;
            theta = 1;
        end
    end
  
 sss = elly0 + grady0'*(x_new - y0); 
    if sss > 1e-10
        alpha_init = alpha + d;
    else
        alpha_init = alpha;
    end
    
    iter = iter + 1;
    
end



