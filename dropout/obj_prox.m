% The objective of proximal mapping
function obj = obj_prox(f, x, X, Xsq, lambda)

  n = size(X, 2); % number of training examples
  d = size(X, 1); % number of features
  discriminant = (f'*X)';
  p = 1 ./ (1 + exp(-discriminant));
  T = repmat(p'.*(1-p'), d, 1) .* Xsq;  
  obj = lambda*norm(f-x)^2/2 + sum(sum(T .* repmat(f.^2, 1, n)))/n;
end
