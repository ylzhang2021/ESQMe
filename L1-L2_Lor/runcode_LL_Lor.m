clear;
clc;

rand('seed',2000);
randn('seed',2023);

maxiter = inf;
tol = 1e-4;
alpha_init = 0.05; % regularization parameter for ESQMe
mu = 0.95;
freq = 1000; % the frequency of print the results

indexarray = [2, 4, 6, 8, 10];      % problem size


repeat = 20;

table1 = [];

% a = clock;
% fname = ['Results\Lorentzian_table' '-'  date '-' int2str(a(4)) '-' int2str(a(5)) '.txt'];
% fid = fopen(fname , 'w');
% fprintf(fid, '%6s & %6s & %6s & %6s (%4s) & %6s (%4s)  & %6s & %6s  & %6s  & %6s & %6s & %6s & %6s \n', ...
%     'index', 't_qr','t_xfeas', 't_scp', 't_sum1', 't_E', 't_sum2', 't_E_r',  'err_scp' , 'err_E', 'err_E_r', 'res_scp', 'res_E', 'res_E_r');

for ii = 1: length(indexarray)
    index = indexarray(ii);  % problem size
    m = 720 * index;
    n = 2560 * index;
    k = 80 * index;     % sparsity of xorig

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

        nf = 0.01;
        error = nf*tan(pi*(rand(m,1) - 1/2));
        b = A*xorig + error;

        gamma = 0.05;
        tmpvalue =  sum(log(1 + error.^2/gamma^2));
        sigma = 1.1* tmpvalue;
        if sigma >= sum(log(1 + b.^2/gamma^2))
            error('0 is included in the feasible set. \n');
        end

        tstart0  = tic;
        [Q,R] = qr(A',0);
        time_qr = toc(tstart0);

        tstart1 = tic;
        xfeas1 = Q*(R'\b);       % taking A^{dagger}b as the initial point for SCP_{ls}_{LL2}
        time_xf1= toc(tstart1);

        M = (norm(xfeas1, 1) - mu*norm(xfeas1))/(1 - mu); % Upper bound of x for ESQM and ESQMe
        x0 = zeros(n,1); % The initial point for ESQM and ESQMe

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

        fprintf('time_qr = %6.4f ; time_xf1 = %6.4f; time_LA = %6.4f \n', time_qr, time_xf1,time_LA);

        LA = nmA;


        fprintf('\n Start of SCP_ls_LL2 \n');
        tstart3 = tic;
        [ x_scp, iter_scp, talg] = SCP_ls_LL2( A, b, sigma, 1, xfeas1, mu, gamma, tol, maxiter);
        time_scp = toc(tstart3);
        fval_x_scp = norm(x_scp, 1) - mu*norm(x_scp);
        Rec_err_scp = norm(x_scp - xorig)/max(1, norm(xorig));
        Residual_scp = (sum(log(1 + (A*x_scp - b).^2/gamma^2)) - sigma)/sigma;
        fprintf(' SCP_ls_LL2 terminated for l1 : time = %g, iter = %d,  nnz = %g,  fval = %18.12f, rec_err = %g, residual = %g,  \n',...
            time_scp, iter_scp, nnz(abs(x_scp) > 1e-10), fval_x_scp, Rec_err_scp, Residual_scp);

        d = gamma^2/20;


        fprintf('\n Start of ESQM with zeros as the initial point \n');
        tesqme1 = tic;
        [x_esqme1, iter_esqme1] = L1L2_ESQM_Lor(A, b, sigma, mu, M, x0, d, alpha_init, 2*sqrt(LA)/gamma^2, gamma, freq, tol, maxiter);
        t_esqme1 = toc(tesqme1);

        fval_esqme1= norm(x_esqme1, 1) - mu*norm(x_esqme1);
        Residual_esqme1 = (sum(log(1 + (A*x_esqme1 - b).^2/gamma^2)) - sigma)/sigma;
        RecErr_esqme1 = norm(x_esqme1 - xorig)/max(1, norm(xorig));
        nnz_esqme1 = nnz(abs(x_esqme1) > 1e-10);
        fprintf(' ESQM_d Termination: time = %g, iter = %d, nnz = %g, fval = %18.12f, rec_err = %g, residual = %g,\n',...
            t_esqme1,  iter_esqme1, nnz_esqme1, fval_esqme1, RecErr_esqme1, Residual_esqme1)

        
        fprintf('\n Start of ESQM_ls with zeros as the initial point \n');
        tesqme2 = tic;
        [x_esqme2, iter_esqme2] = L1L2_ESQM_ls_Lor(A, b, sigma, mu, M, x0, d, alpha_init, 2*sqrt(LA)/gamma^2, gamma, freq, tol, maxiter);
        t_esqme2 = toc(tesqme2);

        fval_esqme2= norm(x_esqme2, 1) - mu*norm(x_esqme2);
        Residual_esqme2 = (sum(log(1 + (A*x_esqme2 - b).^2/gamma^2)) - sigma)/sigma;
        RecErr_esqme2 = norm(x_esqme2 - xorig)/max(1, norm(xorig));
        nnz_esqme2 = nnz(abs(x_esqme2) > 1e-10);
        fprintf(' ESQM_ls_d Termination: time = %g, iter = %d, nnz = %g, fval = %18.12f, rec_err = %g, residual = %g,\n',...
            t_esqme2,  iter_esqme2, nnz_esqme2, fval_esqme2, RecErr_esqme2, Residual_esqme2)
    

        fprintf('\n Start of ESQMe with zeros as the initial point \n');
        tesqme3 = tic;
        [x_esqme3, iter_esqme3] = L1L2_ESQMe_Lor(A, b, sigma, mu, M, x0, d, alpha_init, 2*sqrt(LA)/gamma^2, gamma, freq, tol, maxiter);
        t_esqme3 = toc(tesqme3);

        fval_esqme3= norm(x_esqme3, 1) - mu*norm(x_esqme3);
        Residual_esqme3 = (sum(log(1 + (A*x_esqme3 - b).^2/gamma^2)) - sigma)/sigma;
        RecErr_esqme3 = norm(x_esqme3 - xorig)/max(1, norm(xorig));
        nnz_esqme3 = nnz(abs(x_esqme3) > 1e-10);
        fprintf(' ESQMe_d Termination: time = %g, iter = %d, nnz = %g, fval = %18.12f, rec_err = %g, residual = %g,\n',...
            t_esqme3, iter_esqme3, nnz_esqme3, fval_esqme3, RecErr_esqme3, Residual_esqme3)

        %    fprintf(' norm(x_scp - x_esqme1):  x-x1 = %g, x-x2= %g,\n', norm(x_scp - x_esqme1), norm(x_scp - x_esqme2))

        % save the results
        table2 = [table2; time_qr, time_xf1, time_LA, time_scp, t_esqme1, t_esqme2, t_esqme3, iter_scp, iter_esqme1, iter_esqme2,  iter_esqme3, Rec_err_scp, RecErr_esqme1, RecErr_esqme2, RecErr_esqme3, Residual_scp, Residual_esqme1, Residual_esqme2, Residual_esqme3];
    end

    table1 = [table1; mean(table2)];
    % fprintf(fid, ' %6d & %6.3f & %6.3f  & %6.3f (%6.3f) & %6.3f (%6.3f) & %6.3f & %3.2e & %3.2e & %3.2e & %6.4e& %6.4e & %6.4e \n' , index, mean(table2) );
end
%fclose(fid);

% Save the results as columns
table1 = table1';

a = clock;
fname = ['Results\L1-L2_Lor_table' '-'  date '-' int2str(a(4)) '-' int2str(a(5)) '.txt'];
fid = fopen(fname, 'w');

for ii = 1:7
    fprintf(fid, '& %6.3f & %6.3f & %6.3f & %6.3f & %6.3f \n', table1(ii,:));
end
for ii = 8:11
    fprintf(fid, '& %6.0f & %6.0f & %6.0f & %6.0f & %6.0f \n', table1(ii,:));
end
for ii = 12:15
    fprintf(fid, '& %6.3f & %6.3f & %6.3f & %6.3f & %6.3f\n', table1(ii,:));
end
for ii = 16:19
    fprintf(fid, '& %3.2e & %3.2e & %3.2e & %3.2e & %3.2e\n', table1(ii,:));
end

fclose(fid);


