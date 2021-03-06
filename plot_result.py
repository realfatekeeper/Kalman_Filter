import matplotlib.pyplot as plt
import pandas as pd

result = pd.read_csv("result00.csv")
t = result.iloc[:, 0]
plt.scatter(result['x'],result['y'],s = 0.1,c =  'r')
plt.scatter(result['GPSx'],result['GPSy'],s = 0.01,c='b')
plt.axis('equal')
plt.show()