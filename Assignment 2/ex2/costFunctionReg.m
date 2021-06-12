function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% for the regularized theta, make theta0 = 0
shifted_theta = theta(2:size(theta));
regularized_theta = [0;shifted_theta];

predictions = sigmoid(X*theta);
probability = (-y' * log(predictions)) - ((1-y)' * log(1-predictions));
regularization = (lambda/(2*m)) * regularized_theta' * regularized_theta;

J = (1/m) * probability + regularization;

grad = (1/m) * (X' * (predictions-y) + lambda * regularized_theta);


% =============================================================

end
