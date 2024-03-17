import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
data2 = pd.read_csv("data2.csv")
# print(data)

# plt.scatter(data2.X, data2.Y)
# plt.show()

def loss_function(m, b, points):
    total_error = 0
    for i in  range(len(points)):
        x = points.iloc[i].X
        y = points.iloc[i].Y        
        total_error += (y - ( m * x * b)) ** 2
    total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].X
        y = points.iloc[i].Y
        
        m_gradient += -(2/n) * x * (y - (m_now * x * b_now))
        b_gradient += -(2/n) * (y - (m_now * x * b_now))
    
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    
    return m, b

m, b, L = 0, 0, 0.0001
epochs = 1000

for i in range(epochs):
    m, b = gradient_descent(m, b, data2, L)
    
    if epochs % 50 == 0:
        print(f'Epochs = {i}')

print(m, b)
plt.scatter(data2.X, data2.Y, color='black')
plt.plot(list(range(0, 30)), [m * x * b for x in range(0, 30)], color='red')
plt.show()