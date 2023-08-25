function [ x, iter, talg] = SCP_ls_LL2( A,b,delta,alpha,x_start,mu,gamma,tol,maxiter )
%This code is the MBADC solving the model
% min ||x||_1 - mu||x||
% s.t. \|Ax - b\|_{LL2,gamma} - delta\leq 0
% It uses SCP_ls
%
% Input
%
% A                - m by n matrix (m << n)
% b                - m by 1 vector measurement
% delta            - real number > 0
% alpha            - real number > 0
% x_start          - n by 1 vector starting point
% mu               - real number >= 0
% gamma            - real number > 0
% tol              - tolerance
% maxiter          - max number of iteration
%
% Output
%
% x       - approximate stationary point
% iter    - number of iterations
% talg    - cputime
% X       - n by iter matrix with the t_th column being x^t generated in the t_th iteration, t>=1.

tau = 2;
c = 1e-4;
if alpha < c/2
    error('alpha0 should be no less than than c/2 to guarantee (2.5) in SCP_ls to hold trivially.\n')
end

tstart_alg = tic;

x = x_start;
Lg = 1;
Axminusb = A*x - b;
nablag =  A'*(2*Axminusb./(gamma^2 + Axminusb.^2));
[~,n] = size(A);
iter = 0;

while iter <= maxiter
    %compute partial P_2
    nrmx = norm(x);
    if nrmx <= 1e-8
        xi = zeros(n,1);
    else
        xi = mu*x/nrmx;
    end
    
    %compute subproblem  ( Step (3) )
    y = x + xi/alpha;
    s = x - nablag/Lg;
    r = (norm(nablag)/Lg)^2 - 2/Lg*( sum(log(1 + Axminusb.^2/gamma^2)) - delta);
    [xnew,~] = SubP_alpha(y,s,r,alpha);
    
    Axminusbnew = A*xnew - b;
    
    %Step 3a)
    while sum(log(1 + Axminusbnew.^2/gamma^2) ) > delta
        Lg = tau*Lg; %linesearch
        %compute subproblem  ( Step (3) ) again
        s = x - nablag/Lg;
        r = (norm(nablag)/Lg)^2 - 2/Lg*( sum(log(1+ Axminusb.^2/gamma^2)) - delta);
        [xnew,~] = SubP_alpha(y,s,r,alpha);

        Axminusbnew = A*xnew - b;
    end
    
    %termination criterion
    if norm(x - xnew) < tol*max(norm(xnew),1)
        x = xnew;
        break
    end
    
    
    % preparing for next iteration
    %%update initial stepzise
    
    nablagnew = A'*(2*Axminusbnew./(gamma^2+Axminusbnew.^2));
    Deltag = nablagnew - nablag;
    tmp = (xnew - x)'*Deltag;
    if tmp >= 1e-12
        Lbb = max(1e-8,min(1e8,tmp/norm(xnew - x)^2));
    else
        Lbb = max(1e-8,min(1e8,Lg/tau));
    end
    Lg = Lbb;
    %%update x
    x = xnew;
    Axminusb = Axminusbnew;
    nablag = nablagnew;
    iter = iter + 1;    
end


talg = toc(tstart_alg);
end

