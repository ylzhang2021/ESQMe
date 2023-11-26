function [xstar, lambda] = Newton_Monotone_LL(y, a, sigma, alpha, lambda, M, tol, L)
% This aims to find the root of T(lambda) = 0 with s = 0;
% where T(lambda) = sigma - a^T*x;
% and x = sign(tmp)*min(max(abs(tmp) - gamma, 0), M);
% and tmp = y - lambda*gamma*a.

% Input
%
% y             - n by 1 vector measurement
% a              - n by 1 vector measurement
% sigma      - real number > 0
% alpha     - real number > 0
% lambda   - real number which is the initial of lambda
% M           - real number > 0
% tol            - tolerance[1e-10] 
% L             - a constant [ \|A\|^2 ]
%
%
% Output
%
% xstar       - approximate stationary point
% lambda   - corresponding Lagrange multipliers

% Initialization
eta = 1e-4; % parameter for regular
c = 1e-4;
iter = 0;

% Compute function value
tmp = y - (lambda/(L*alpha)).* a;
xstar = sign(tmp).*min(max(abs(tmp) - (1/(L*alpha)), 0), M);
g = sigma - a'*xstar;
normg = abs(g);

while normg > tol
    
    muk = eta*normg^(1/2);
    
    %  calculate the generalized Jacobian
    I = abs(tmp) > (1/(L*alpha)) & abs(tmp) <= (1/(L*alpha)) +M;  %  gamma< \|y_J\| <= gamma + M
    a0 = a(I);
    H = (1/(L*alpha)).*norm(a0)^2;
    
    % Calculate the direction
    H = H + muk;
    dir = H\(-g);
    
    % Linesearch
    s = 1;
    while 1==1
        u = lambda + s*dir;
        tmp = y - (u/(L*alpha)).*a;
        xstar = sign(tmp).*min(max(abs(tmp) - (1/(L*alpha)), 0), M);
        g = sigma - a'*xstar;
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

end
