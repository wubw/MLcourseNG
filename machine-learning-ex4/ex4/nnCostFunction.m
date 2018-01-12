function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];
A2 = sigmoid(X * Theta1');
A2 = [ones(m, 1) A2];
A3 = sigmoid(A2 * Theta2');
sum = 0;

for i = 1:m
    for k = 1:num_labels
        yik = (y(i) == k);
        sum += -1 * yik * log(A3(i, k)) - (1 - yik) * log(1 - A3(i, k));
    end
end

reg_sum = 0;

for j = 1:hidden_layer_size
    for k=1:input_layer_size
        reg_sum += (Theta1(j, k+1) * Theta1(j, k+1));
    end
end

for j = 1:num_labels
    for k=1:hidden_layer_size
        reg_sum += (Theta2(j, k+1) * Theta2(j, k+1));
    end
end

J = sum / m + reg_sum * lambda / 2 / m;

yy = zeros(m, num_labels);
for i = 1:m
    yy(i, y(i)) = 1;
end

delta3 = A3 .- yy;
delta2 = (delta3 * Theta2) .* A2 .* (1-A2);
delta2 = delta2(1:end, 2:end);
Theta1_grad = Theta1_grad .+ (delta2' * X);
Theta2_grad = Theta2_grad .+ (delta3' * A2);

for i = 1:hidden_layer_size
    for j = 1:(input_layer_size + 1)
        if j != 1
            Theta1_grad(i, j) = Theta1_grad(i, j) .+ (Theta1(i, j) * lambda);
        end
    end
end

for i = 1:num_labels
    for j = 1:(hidden_layer_size + 1)
        if j != 1
            Theta2_grad(i, j) = Theta2_grad(i, j) .+ (Theta2(i, j) * lambda);
        end
    end
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
