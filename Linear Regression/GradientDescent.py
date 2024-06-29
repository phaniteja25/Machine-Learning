import numpy as np
def gradient_descent(x,y):
    m_curr = b_curr = 0 #m_curr is the slope and b_curr is the y intercept
    iterations = 1000
    n = len(x)
    learning_rate = 0.08
    for i in range(iterations):
        print(x)
        print(y)
        y_predicted = m_curr*x + b_curr
        print(y_predicted)
        m_derivative = -(2/n) * sum(x*(y-y_predicted))
        b_derivative = -(2/n) * sum(y-y_predicted)
        m_curr = m_curr - m_derivative*learning_rate
        b_curr = b_curr - b_derivative*learning_rate
        cost_function  = (1/n) * sum([val**2 for val in (y-y_predicted)])
        print("m {},b {}, iteration {}, cost: {} ".format(m_curr,b_curr,i,cost_function))
    pass


x = np.array([1,2,3,4,5]) #)independent
y = np.array([5,7,9,11,13]) #dependent

gradient_descent(x,y)