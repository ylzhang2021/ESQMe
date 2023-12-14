clc
i = 0;
theta0 = 1;
theta1 = 1;
while 1 == 1
    beta = (theta0 - 1)/theta1; % this is beta_i
    fprintf('k = %d, beta_k = %6.5f, fine if < 0: %d\n',[i beta sign(beta-sqrt(2/2.25))])
    if beta >= sqrt(2/2.25)
        break
    end
    theta0 = theta1;
    theta1 = (1 + sqrt(1+4*theta0^2))/2;
    i = i+1;
end