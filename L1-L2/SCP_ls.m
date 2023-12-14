function [ x, iter, talg] = SCP_ls( A,b,delta,alpha,x_start,mu,tol,maxiter )
%This code uses a variant of the SCP method solving the model
% min ||x||_1 - mu*||x||
% s.t. 1/2*||Ax - b||^2 - delta <=0
% It uses SCP_ls
%
% Input
%
% A                - m by n matrix (m << n)
% b                - m by 1 vector measurement
% delta            - real number > 0
% alpha            - real number > 0
% x_start          - n by 1 vector starting point
% mu               - real number > 0
% tol              - tolerance
% maxiter          - max number of iteration
% 
% Output
%
% x       - approximate stationary point
% iter    - number of iterations
% talg    - cputime


tau = 2;
c = 1e-4;
if alpha<c/2
    error('alpha0 should be no less than than c/2 to guarantee (2.5) in SCP_ls holds trivially.\n')
end


tstart_alg = tic;

Lg = 1;
x = x_start;
Ax = A*x;
nablag = A'*(Ax - b); %g(x)µÄÌÝ¶È
[~,n] = size(A);
iter = 0;



while iter<=maxiter
    %compute partial P_2
    nrmx = norm(x);
    if nrmx <= 1e-8
        xi = zeros(n,1);
    else
        xi = mu*x/nrmx;
    end   
    
    %compute subproblem  ( Step (3) )
    y = x + xi/alpha;
    temp = Ax - b;
    s = x - nablag/Lg;
    r = (norm(nablag)/Lg)^2 - 2/Lg*(1/2*(norm(temp))^2 - delta);
    
    [xnew,~] = SubP_alpha(y,s,r,alpha);
    Axnew = A*xnew;
    
    %Step 3a)
    while  1/2*norm(Axnew - b)^2 - delta >0 %step3a)
        Lg = tau*Lg; %linesearch
        %compute subproblem  ( Step (3) ) again
        s = x - nablag/Lg;
        r = (norm(nablag)/Lg)^2 - 2/Lg*(1/2*(norm(temp))^2 - delta);
        
        [xnew,~] = SubP_alpha(y,s,r,alpha);
        Axnew = A*xnew;
    end
    nablagnew = A'*(Axnew - b);
      
    %termination criterion
    if norm(x - xnew) < tol*max(norm(xnew),1)
        x = xnew;
        break
    end
    
    % preparing for next iteration
    %%update initial stepsize in the next iteration
    Deltag = nablagnew - nablag;    
    tmp = (xnew - x)'*Deltag;
    if tmp >= 1e-12
        Lbb = max(1e-8,min(1e8,tmp/norm(xnew - x)^2));
    else
        Lbb = max(1e-8,min(1e8,Lg/tau)); % Edited this on July 14, 2020
    end
    Lg = Lbb;
    %%update x
    x = xnew;
    Ax = Axnew;
    nablag = nablagnew;
    
    iter = iter + 1;

end


talg = toc(tstart_alg);
end

