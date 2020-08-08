% Main file of the experiment for dropout
% \lambda and \mu can be set to a range in line 16 and 17
% c = factor*lambda^2/2*mu (by the last equation on page 4)
% So users can set the value of 'factor' in line 20

load 2moons;
X = x';
d = size(X, 1); % #features
n = size(X, 2); % #training examples
try_dropout = 1;
try_prox = 1;

% n = 10;   % uncomment to take a subset of the dataset
X = X(:, 1:n);
y = y(1:n);

Xsq = X.^2;
X_new = X;
options = optimoptions(@fminunc,'Display','none');

for lambda = 0.5 %:0.4:0.6
  for mu = 0.1
    for factor = 0.1:0.1:0.4

      c = factor*lambda^2/2*mu;

      if try_dropout
        opt_obj_dropout = @(beta)obj_dropout(beta, X, Xsq, y, mu);        
        beta_dropout = fminunc(opt_obj_dropout, zeros(d, 1), options);
        output_dropout = (beta_dropout' * X)';
      end

      if try_prox
        for i = 1 : n  
          x = X(:,i);
          X_new(:, i) = fminunc(@(f)obj_prox(f, x, X, Xsq, lambda), x, options);
        end
        beta_prox = fminunc(@(beta)obj_logistic(beta, X_new, y, c), zeros(d,1));
        output_prox = (beta_prox' * X)';
      end

      scatter(output_dropout, output_prox);
      m = max(abs([output_dropout; output_prox]));
      axis equal
      range = [-m,m];
      xlim(range); ylim(range);
      hold on;
      plot(range, range);
      box on;
      fname = sprintf('output_%.2f_%.2f_%.2f.jpg', lambda, mu, factor);
      saveas(gcf, fname, 'jpg')
      close all;
    end
  end
end