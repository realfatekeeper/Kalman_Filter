import matplotlib.pyplot as plt
import pandas as pd

history = pd.read_csv("history00.csv")
plt.scatter(history['x'],history['y'],s = 0.1,c='r')
plt.axis('equal')
plt.show()