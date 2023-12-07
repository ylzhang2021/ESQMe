function [x_new, iter] =  L1L2_ESQMb(A, b, delta, mu, M, xstart, d, alpha_init, L, maxiter, freq, tol)
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
% alpha_init   - real number > 0
% L                - the Lipschitz constant
% maxiter      - maximum number of iterations [inf]
% freq            - The frequency of print the results
% tol              - tolerance 


% Output

% x_new      - approximate stationary point
% iter           - number of iterations


% Initialization

lambda = 0; % parameter for subproblem
iter = 0;

x_old = xstart; % starting point x^{k}
x_new = x_old;% starting point x^{x+1}



% fprintf(' ****************** Start   ESQMe ********************\n')
% fprintf('  iter        fval        gvalu      alpha       lambda      norm(x_new - x_old)        norm(x_new)\n')


while  iter <= maxiter
    
    alpha = alpha_init; 
    
    if norm(x_new) <= 1e-10
    xi = 0*x_new;
    else
    xi = mu*x_new/norm(x_new);
    end
    
    % iterations, gradient    
    Ax_new = A * x_new;
    tmpx_new = Ax_new - b;
    gvalx_new = (1/2)*norm(tmpx_new)^2 - delta;
    gradx_new = A'*tmpx_new;
    y = x_new + (1/(L*alpha)).*xi;   
    
    x_old = x_new;

    % Solving the subproblem
    [x_new, lambda] = subprob_ESQM(y, gradx_new, gradx_new'*x_new - gvalx_new, alpha, lambda, M, L);

     
%     if mod(iter, freq) == 0
%         fprintf(' %5d| %16.10f|%3.3e|%3.3e|%3.3e|%3.3e|%3.3e\n',iter, norm(x_new, 1) - mu*norm(x_new), gvalx_new, alpha, lambda, norm(x_new - x_old),  norm(x_new) )
%     end
     
     % check for termination

     if norm(x_new - x_old) < tol*max(norm(x_new), 1)
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



