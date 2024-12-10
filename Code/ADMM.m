%Test Data
A = [ 1, 2, 3;
     4, 5, 6
     7, 8, 10;];    % A 3x3 matrix
b = [1; 2; 3];     % Target vector (dimension 3x1)
mu = 0.5;          % Regularization parameter
lambda = 1.0;      % Augmented Lagrangian parameter

tol = 1e-6;      % Convergence tolerance
max_iter = 1000; % Maximum number of iterations
x_prev = x;              % To track previous x
y_prev = y;              % To track previous y
% Initialization
[d, ~] = size(A);        % Dimension of x
x = zeros(d, 1);         % Initialize x
y = zeros(d, 1);         % Initialize y
z = zeros(d, 1);         % Initialize dual variable z
obj_values = [];         % To store the objective function values
% ADMM iterations
for k = 1:max_iter
    % Step 1: x-update
    x = (lambda * eye(d) + A' * A) \ (A' * b + lambda * y - z);

    % Step 2: y-update (proximal operator of l1-norm)
    v = x + (z / lambda);
    y = sign(v) .* max(abs(v) - mu / lambda, 0);

    % Step 3: z-update (dual variable update)
    z = z + lambda * (x - y);
    obj = mu * norm(y, 1) + 0.5 * norm(A * x - b, 2)^2;
    obj_values = [obj_values; obj];
    % Check for convergence
    if norm(x - y, 2) < tol && norm(y - y_prev, 2) < tol
        fprintf('Converged at iteration %d\n', k);
        break;
    end

    % Update previous x and y
    x_prev = x;
    y_prev = y;
end

% Results
fprintf('Optimal x:\n');
disp(x);
fprintf('Optimal y:\n');
disp(y);
fprintf('Optimal dual variable z:\n');
disp(z);

