import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def using_sklearn():
    df = pd.read_csv("test_scores.csv")
    lg_model = LinearRegression()
    lg_model.fit(df[["math"]],df.cs)
    return lg_model.coef_,lg_model.intercept_


def gradient_descent(x,y):
    m_curr = 0
    b_curr = 0
    learning_rate = 0.0001
    n = len(x)
    prev_cost=0
    iterations = 100000
    for i in range(iterations):
        y_predicted = m_curr*x + b_curr
        cost_function = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        m_derivative = -(2/n)*sum(x*(y-y_predicted))
        b_derivative = -(2/n)*sum((y-y_predicted))
        m_curr = m_curr - learning_rate*m_derivative
        b_curr= b_curr - learning_rate*b_derivative

        if(math.isclose(cost_function,prev_cost,rel_tol=1e-20)):
            break
        prev_cost = cost_function

    print("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr, cost_function, i))

    return m_curr,b_curr



if __name__ == "__main__":
    df = pd.read_csv("test_scores.csv")
    x = np.array(df.math)
    y = np.array(df.cs)

    slope,intercept = gradient_descent(x,y)
    print("Using Gradient Descent Slope : {}, Intercept : {}".format(slope,intercept))

    slope2, intercept2 = using_sklearn()
    print("Using Built In : {}, Intercept : {}".format(slope2, intercept2))