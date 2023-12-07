clear;
clc;

rand('seed',2000);
randn('seed',2023);

maxiter = inf;
tol = 1e-4;
d = 1;
alpha_init = 1;
mu = 0.95;
freq = 10000; % the frequency of print the results

indexarray = [2, 4, 6, 8, 10];      % problem size

repeat = 20;

table1 = [];

% a = clock;
% fname = ['Results\Normal_table' '-'  date '-' int2str(a(4)) '-' int2str(a(5)) '.txt'];
% fid = fopen(fname, 'w');
% fprintf(fid, '%6s &%6s &%6s &%6s & %6s (%4s) &%6s (%4s)  & %6s (%4s) & %6s & %6s  & %6s & %6s & %6s & %6s \n', ...
%     'index', 't_qr', 't_lambda', 't_xfea','t_scp', 't_sum', 't_esqme', 't_sum1', 't_esqme_r', 't_sum2', 'err_scp' ,  'err_esqme', 'err_esqme_r', 'res_scp', 'res_esqm','res_esqm_r');

for ii = 1: length(indexarray)
    index = indexarray(ii);  % problem size
    m = 720 * index;
    n = 2560 * index;
    k = 160 * index;     % sparsity of xorig
    
     table2 = [];

    for rr = 1 : repeat
        [index, rr]
        % generate the original signal
        I = randperm(n);
        J = I(1:k);
        xorig = zeros(n,1);
        xorig(J) = randn(k,1);
        
        A = randn(m,n);
        for j = 1:n
            A(:,j) = A(:,j)/norm(A(:,j));
        end
        error = 0.01*randn(m, 1);  % Gaussian noise
        b = A*xorig + error;
        
        sigma = 1.1* norm(error);   
        delta = sigma^2/2;
        
        tstart0  = tic;
        [Q,R] = qr(A',0);
        time_qr = toc(tstart0);
        
        tstart1 = tic;
        xfeas1 = Q*(R'\b);       % Taking A^{dagger}b as the initial point for SCP_ls
        time_xf1= toc(tstart1);

        x0 = zeros(n, 1); %Taking zeros as the initial point for ESQM and ESQMe
        M =  (norm(xfeas1, 1) - mu*norm(xfeas1))/(1 - mu); % Upper bound of x for ESQM and ESQMe
       
        
        % Calculate the Lipschitz constant
        if m > 2000
            clear opts
            opts.issym = 1;
            tstart = tic;
            nmA = eigs(A*A', 1, 'LM');
            time_LA = toc(tstart);
        else
            tstart = tic;
            nmA = norm(A*A');
            time_LA = toc(tstart);
        end
        fprintf(' Lipschitz constant L = %g\n', nmA)
        
         fprintf('time_qr = %6.4f ; time_xf1 = %6.4f; time_lambda = %6.4f \n', time_qr, time_xf1, time_LA);

        LA = nmA;
        
        
        fprintf('\n Start of SCP_ls \n');
        tstart3 = tic;
        [x_scp, iter_scp, talg] = SCP_ls( A, b, delta, 1, xfeas1, mu, tol, maxiter);
        time_scp = toc(tstart3);
        
        fval_x_scp= norm(x_scp, 1) - mu*norm(x_scp); 
        Rec_err_scp = norm(x_scp - xorig)/max(1, norm(xorig));
        Residual_scp = (1/2*norm(A*x_scp - b)^2 - delta)/delta;
        fprintf(' SCP_ls terminated : time = %g, iter = %d,  nnz = %d,  fval = %16.10f, rec_err = %g, residual = %7.4e  \n',...
            time_scp, iter_scp, nnz(abs(x_scp) > 1e-10), fval_x_scp, Rec_err_scp, Residual_scp);
        
       
        fprintf('\n Start of ESQMb with zeros as the initial point \n');
        tesqme1 = tic;
        [x_esqme1, iter_esqme1] = L1L2_ESQMb(A, b, delta, mu, M, x0, d, alpha_init, LA, maxiter, freq, tol);
        t_esqme1 = toc(tesqme1);
        
        fval_esqme1= norm(x_esqme1, 1) - mu*norm(x_esqme1); 
        Residual_esqme1 = (1/2*norm(A*x_esqme1 - b)^2 - delta)/delta; 
        RecErr_esqme1 = norm(x_esqme1 - xorig)/max(1, norm(xorig));
        nnz_esqme1 = nnz(abs(x_esqme1)  > 1e-10);
        fprintf(' ESQMb  terminated: time = %g, iter = %d, nnz = %g, fval = %16.10f, diff = %g, feas vio = %g,\n',...
            t_esqme1, iter_esqme1, nnz_esqme1, fval_esqme1, RecErr_esqme1, Residual_esqme1)
       
             
        % The initial point is random
        fprintf('\n Start of ESQMe with zeros as the initial point \n');
        tesqme2 = tic;
        [x_esqme2, iter_esqme2] = L1L2_ESQMe(A, b, delta, mu, M, x0, d, alpha_init, LA, maxiter, freq, tol);
        t_esqme2 = toc(tesqme2);
        
        fval_esqme2= norm(x_esqme2, 1) - mu*norm(x_esqme2); 
        Residual_esqme2 = (1/2*norm(A*x_esqme2 - b)^2 - delta)/delta; 
        RecErr_esqme2 = norm(x_esqme2 - xorig)/max(1, norm(xorig));
        nnz_esqme2 = nnz(abs(x_esqme2)  > 1e-10);
        fprintf(' ESQMe  terminated: time = %g, iter = %d, nnz = %g, fval = %16.10f, diff = %g, feas vio = %g,\n',...
            t_esqme2, iter_esqme2, nnz_esqme2, fval_esqme2, RecErr_esqme2, Residual_esqme2)
    
     % save the results
        table2 = [table2; time_qr, time_xf1, time_LA, time_scp, t_esqme1, t_esqme2, iter_scp,  iter_esqme1, iter_esqme2, Rec_err_scp, RecErr_esqme1, RecErr_esqme2, Residual_scp, Residual_esqme1, Residual_esqme2];
    
    end

    table1 = [table1; mean(table2)];
    
%     fprintf(fid, '%6d & %6.4f & %6.4f & %6.4f & %6.4f (%6.4f)  & %6.4f (%6.4f) & %6.4f  (%6.4f) & %6.4e & %6.4e & %6.4e & %6.4e & %6.4e& %6.4e \n' , index, mean(table2) );
end
% fclose(fid);

% Save the results as columns
table1 = table1';

a = clock;
fname = ['Results\L1-L2_1e-4_table' '-'  date '-' int2str(a(4)) '-' int2str(a(5)) '.txt'];
fid = fopen(fname, 'w');

for ii = 1:6
    fprintf(fid, '& %6.3f & %6.3f & %6.3f & %6.3f & %6.3f \n', table1(ii,:));
end
for ii = 7:9
    fprintf(fid, '& %6.0f & %6.0f & %6.0f & %6.0f & %6.0f \n', table1(ii,:));
end
for ii = 10:12
   fprintf(fid, '& %6.3f & %6.3f & %6.3f & %6.3f & %6.3f\n', table1(ii,:));
end
for ii = 13:15
    fprintf(fid, '& %3.2e & %3.2e & %3.2e & %3.2e & %3.2e\n', table1(ii,:)); 
end

fclose(fid);
 

