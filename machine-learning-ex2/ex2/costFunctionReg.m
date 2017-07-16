function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
temp1 = -1 * (y .* log(sigmoid(X * theta)));
temp2 = (1 - y) .* log(1 - sigmoid(X * theta));

thetaT = theta;
thetaT(1) = 0;
correction = sum(thetaT .^ 2) * (lambda / (2 * m));

J = sum(temp1 - temp2) / m + correction;

grad = (X' * (sigmoid(X * theta) - y)) * (1/m) + thetaT * (lambda / m);

end
