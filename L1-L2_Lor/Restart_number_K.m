beta = 0;
i = 0;
theta0 = 1;
theta1 = 1;
while beta < sqrt(2/2.25)
    beta = (theta0 - 1)/theta1
    theta0 = theta1;
    theta1 = (1 + sqrt(1+4*theta0^2))/2;
    i = i+1;
end
i
beta