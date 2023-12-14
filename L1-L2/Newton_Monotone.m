function [xstar, lambda] = Newton_Monotone(y, a, sigma, alpha, lambda, M, tol)
% This aims to find the root of T(lambda) = 0 whenever s = 0;
% where T(lambda) = sigma - a^T*x; 
% and x = min(max(1 - 1/alpha./|tmp|,0), M./|tmp|).*tmp ;
% and tmp = y - (lambda/alpha).*a.

% Input

% y              - n by 1 vector measurement
% a              - n by 1 vector measurement
% sigma       - real number > 0
% alpha        - real number > 0
% lambda     - real number which is the initial of lambda
% M             - real number > 0
% L              - the Lipschitz constant
% tol              - tolerance [1e-10]



% Output

% xstar       - approximate solution point
% lambda   - corresponding Lagrange multipliers


% Initialization
eta = 1e-4; 
c = 1e-4;

iter = 0;

% Compute function value
tmp = y - (lambda/alpha).*a; 
abstmp = abs(tmp);
xstar = sign(tmp).*min(max(abstmp - 1/alpha, 0), M); 
g = sigma - sum(a.*xstar); 
normg = abs(g);


if normg <= tol % Check  lambda = the initial lambda?
    return
else    
    while normg > tol

        muk = eta*normg^(1/2);
        
        %  Calculate the generalized Jacobian
        I = abstmp > 1/alpha & abstmp <= 1/alpha +M;  % Only need to consider the cases gamma< \|y_i\| <= gamma + M
        H = (1/alpha)*sum(a(I).^2);
        
        % Calculate the direction
        H = H + muk;
        dir = H\(-g);
        
        % Linesearch 
        s = 1;
        while 1 == 1            
            u = lambda + s*dir;
            tmp = y - (u/alpha)*a;
            abstmp = abs(tmp);
            xstar = sign(tmp).*min(max(abstmp - 1/alpha, 0), M); 
            g = sigma - sum(a.*xstar);
            if g*dir + c*muk*dir^2 <= 0 || s < 1e-10
                break
            else
                s = s/2;
            end
        end

        lambda = u;
        normg = abs(g);
        
        if s < 1e-10
            break
        end
        iter = iter + 1;
    end
%     normg
end
 