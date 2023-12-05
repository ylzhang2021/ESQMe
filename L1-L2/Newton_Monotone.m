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
        I = abstmp > 1/alpha & abstmp <= 1/alpha +M;  %  gamma< \|y_J\| <= gamma + M
        tmpmat = a(:,I);
        tmp1 = tmp(:,I);
        abstmp1 = abstmp(I);
        H1 = (1/alpha)^2*sum(sum((tmp1.*tmpmat)).^2./(abstmp1.^3)) + (1/alpha)*sum((1 - (1/alpha)./abstmp1).*sum(tmpmat.^2));
        
        K = abstmp > (1/alpha) + M;  % \|y_J\| > gamma + M
        tmpmat1 = a(:,K);
        tmp11 = tmp(:,K);
        abstmp11 = abstmp(K);
        H2 = - M*(1/alpha)*sum(sum((tmp11.*tmpmat1)).^2./(abstmp11.^3)) + M*(1/alpha)*sum((1./abstmp11).*sum(tmpmat1.^2));
        
        H = H1 + H2;
        
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
 