import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df=pd.read_csv('test_scores.csv')
df
x=df['math']
x
print(type(x))
y =df['cs']
y
def gradient_descendent(x,y):
    m_curr=b_curr=0
    n=len(x)
    learning_rate=0.0001
    tolerance=1*10**-7
    prev_cost=float('inf')
    iterations=0
    m_values=[]
    b_values=[]
    cost_values=[]    
    
    while True:
        y_predicted=m_curr*x+b_curr
        cost=(1/n)*sum([val**2 for val in (y-y_predicted)])
        m_prime=(-2/n)*sum(x*(y-y_predicted))
        b_prime=(-2/n)*sum((y-y_predicted))
        m_curr=m_curr-learning_rate*m_prime
        b_curr=b_curr-learning_rate*b_prime
        m_values.append(m_curr)
        b_values.append(b_curr)
        cost_values.append(cost)
        if iterations>0:
            cost_diff=abs(cost-prev_cost)
            print(f'Iteration no: {iterations},Cost is: {cost}Cost Difference :{cost_diff},The slope is:{m_curr},The intercept is:{b_curr}')
            if math.isclose(cost,prev_cost,rel_tol=tolerance):
                print(f'Convergence is achieved at iteration no {iterations},Final cost is :{cost}')
                break
    
        prev_cost=cost
        iterations +=1
    
    print(f'The final parameters are, m:{m_curr},b:{b_curr}')
    return m_curr,b_curr,cost_values,m_values,b_values
    
m_grad,b_grad,costs,m_values,b_values= gradient_descendent(x,y)
x_reshaped = x.to_numpy().reshape(-1, 1)

model=LinearRegression()
model.fit(x_reshaped,y)
m_sklearn=model.coef_[0]
b_sklearn=model.intercept_
print(f'Sklearn LRM results are,m : {m_sklearn}, b: {b_sklearn}')
plt.Figure(figsize=(12,6))
plt.scatter(x,y,color='blue',label='Data Points')
y_grad=m_grad*x+b_grad
plt.plot(x,y_grad,color='green',label='Gradient Descentdent ALgorithm line')
y_sklearn=m_sklearn*x+b_sklearn
plt.plot(x,y_sklearn,color='red',linestyle='dashed',label='Sklearn Regression Line')


for i in range(len(m_values)):
    y_step=m_values[i]*x+b_values[i]
    plt.plot(x,y_step,color='lightgreen')

plt.title('Comparison of Sklearn Linear Regression Model and User Defined Gradient Descendent Algorithm')
plt.xlabel('Math Scores')
plt.ylabel('Cs Scores')
plt.legend()
plt.show()
