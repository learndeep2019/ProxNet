% The objective of logistic regress
function f = obj_logistic(beta, X, y, c)

  n = size(X, 2); % number of training examples
  discriminant = (beta'*X)';
  r = exp(-y.*discriminant);
  
  f = sum(log(1+r))/n + c*(beta'*beta)/2;
end
