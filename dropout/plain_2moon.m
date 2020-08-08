% Plot the original dataset
load('2moons.mat');
%x = X;
%y = Y;
data1 = x(y==1,:);
data2 = x(y==-1,:);
[num1,~] = size(data1);
[num2,~] = size(data2);
data1 = data1(1:num1,:);
X1=data1(:,1);
Y1 = data1(:,2);
X2 = data2(:,1);
Y2 = data2(:,2);
figure(1);
plot(X1, Y1, 'r^', 'MarkerSize',10);
hold on
plot(X2, Y2, 'ks', 'MarkerSize',10);
xlim([-1.5 3])
ylim([-0.8 1.3])
