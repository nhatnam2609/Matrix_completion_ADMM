function test_matrix_completion()
    m = 50;  % rows
    n = 40;  % columns
    r = 3;   % rank
    
    % Create low-rank matrix
    U = randn(m, r);
    V = randn(n, r);
    B_true = max(U * V', 0);  % True nonnegative matrix
    
    % Sample entries randomly
    sample_rate = 0.5;
    Omega = rand(m, n) < sample_rate;
    B = B_true .* Omega;  % Observed matrix
    
    % Algorithm parameters
    mu = 0.1;
    lambda = 2.0;
    max_iter = 10000;
    tol = 1e-6;
    
    % Run ADMM
    [X, Y, Z] = matrix_completion_admm(B, Omega, mu, lambda, max_iter, tol);
    
    % Evaluate results
    rel_error = norm(X - B_true, 'fro') / norm(B_true, 'fro');
    fprintf('Relative error: %e\n', rel_error);
    fprintf('Rank of solution: %d\n', rank(X, 1e-6));
    
    % Visualize results
    figure('Position', [100 100 1200 400]);
    
    subplot(1,3,1);
    imagesc(B_true);
    title('True Matrix');
    colorbar;
    
    subplot(1,3,2);
    imagesc(B);
    title('Observed Matrix (with missing entries)');
    colorbar;
    
    subplot(1,3,3);
    imagesc(X);
    title('Recovered Matrix');
    colorbar;
end