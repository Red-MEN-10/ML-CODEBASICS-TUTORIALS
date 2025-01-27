import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load your dataset
df = pd.read_csv('test_scores.csv')
x = df['math']
y = df['cs']

# Gradient Descent Function
def gradient_descendent(x, y):
    m_curr = b_curr = 0
    n = len(x)
    learning_rate = 0.00038
    tolerance = 1e-6  # Adjusted tolerance
    prev_cost = float('inf')
    iterations = 0
    
    m_values = []  # Store m values for visualization
    b_values = []  # Store b values for visualization
    costs = []  # Store cost values for visualization
    
    while True:
        y_predicted = m_curr * x + b_curr
        cost = (1 / n) * sum([(val ** 2) for val in (y - y_predicted)])
        m_prime = (-2 / n) * sum(x * (y - y_predicted))
        b_prime = (-2 / n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * m_prime
        b_curr = b_curr - learning_rate * b_prime
        
        # Store values for visualization
        m_values.append(m_curr)
        b_values.append(b_curr)
        costs.append(cost)
        
        if iterations > 0:
            cost_diff = abs(cost - prev_cost)
            print(f'Iteration no: {iterations}, Cost: {cost}, Cost Difference: {cost_diff}, Slope: {m_curr}, Intercept: {b_curr}')
            if math.isclose(cost, prev_cost, rel_tol=tolerance):
                print(f'Convergence is achieved at iteration no {iterations}, Final cost is: {cost}')
                break
        
        prev_cost = cost
        iterations += 1
    
    print(f'The final parameters are, m: {m_curr}, b: {b_curr}')
    return m_curr, b_curr, costs, m_values, b_values

# Run Gradient Descent
m_grad, b_grad, costs, m_values, b_values = gradient_descendent(x, y)

# Fit with sklearn for comparison
x_reshaped = x.values.reshape(-1, 1)  # sklearn expects 2D arrays
model = LinearRegression()
model.fit(x_reshaped, y)
m_sklearn = model.coef_[0]
b_sklearn = model.intercept_

print(f"Sklearn Linear Regression Results: m = {m_sklearn}, b = {b_sklearn}")

# Visualization
plt.figure(figsize=(12, 6))

# Scatter plot of data
plt.scatter(x, y, color='blue', label='Data Points')

# Plot gradient descent regression line
y_grad = m_grad * x + b_grad
plt.plot(x, y_grad, color='green', label='Gradient Descent Line')

# Plot sklearn regression line
y_sklearn = m_sklearn * x + b_sklearn
plt.plot(x, y_sklearn, color='red', linestyle='dashed', label='Sklearn Line')

# Gradient descent steps visualization
for i in range(len(m_values)):
    y_step = m_values[i] * x + b_values[i]
    plt.plot(x, y_step, color='lightgreen', alpha=0.2)

plt.title('Comparison of Gradient Descent and Sklearn Linear Regression')
plt.xlabel('Math Scores')
plt.ylabel('CS Scores')
plt.legend()
plt.show()

# Optional: Cost reduction plot
plt.figure(figsize=(8, 5))
plt.plot(range(len(costs)), costs, label='Cost Over Iterations', color='purple')
plt.title('Cost Reduction During Gradient Descent')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.legend()
plt.show()
