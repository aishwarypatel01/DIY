data = table2array([readtable("data_close.xlsx");readtable("data_open.xlsx")]);

X = data(:, [1,2]);
y = data(:, 3);

initial_theta = [0;0];

options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 100);

[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);


p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 300);


filename = "data.xlsx";

writematrix([1;theta], filename);

sigmoid([1 240]*theta)

plot(X(:,2), sigmoid(X*theta))

clear
