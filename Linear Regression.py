import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")
# print(data)

plt.scatter(data.X, data.Y)
plt.show()