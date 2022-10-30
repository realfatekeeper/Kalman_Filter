import pandas as pd
import numpy as np
data = pd.read_csv('kalman00.csv')
distance = 0
for k, row in data.iterrows():
    if k == 0:
        last_row = row
        continue
    elif row["GPSx"] != last_row["GPSx"] and row["GPSy"] != last_row["GPSy"]:
        distance += np.sqrt((pow(row["GPSx"]-last_row["GPSx"], 2)+pow((row["GPSy"] - last_row["GPSy"]), 2)))
        last_row = row
print(distance)