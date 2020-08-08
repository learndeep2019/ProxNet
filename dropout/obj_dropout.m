% The objective of dropout
function f = obj_dropout(beta, X, Xsq, y, lambda)

  n = size(X, 2); % number of training examples
  d = size(X, 1); % number of features
  discriminant = (beta'*X)';
  p = 1 ./ (1 + exp(-discriminant));
  r = exp(-y.*discriminant);
  
  f1 = sum(log(1+r));
  T = repmat(p'.*(1-p'), d, 1) .* Xsq;
  f2 = sum(sum( T .* repmat(beta.^2, 1, n)));
  f = (f1 + lambda/2*f2) / n;  
end
