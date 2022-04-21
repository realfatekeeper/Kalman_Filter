import numpy as np
import pandas as pd

df = pd.read_csv('raw_data\HIMU_uturn.csv')
# print(df.columns.values.tolist())
# ROT_BIAS = [0,0,0]
# TRANS_BIAS = [0,0,0]
""" ROT_BIAS = [-3.7034333488898317e-06, -6.277694338380008e-09, 4.123992894451507e-06]
TRANS_BIAS = [0.03137413793138769, 0.006497261588525887, 0.03793959105307] """

EARTH_RADIUS = 6371000
PI = 3.14159265

# adjust the time
adjusted_df = pd.DataFrame()
adjusted_df['time'] = (df['timestamp'] - df['timestamp'][0])/1000
df["GPS.lat"]
# Calculate location from GPS
xs, ys = df["GPS.lat"],df["GPS.long"]
phi = np.radians(xs) # phi in formula
ld = np.radians(ys) # lambda in formula
x = EARTH_RADIUS * ld
y = EARTH_RADIUS * np.log(np.tan(PI/4 +phi/2))

x = x - x[0]
y = y - y[0]
a = np.sqrt((df["icm4x6xx_accelerometer.x"]+9.81)**2 + df["icm4x6xx_accelerometer.y"]**2+df["icm4x6xx_accelerometer.z"]**2)
adjusted_df['GPSx'] = x
adjusted_df['GPSy'] = y
adjusted_df['a'] = a
adjusted_df['yaw'] = np.radians(df["orientation.x"])
adjusted_df['ax'] = a*np.cos(adjusted_df['yaw'])
adjusted_df['ay'] = a*np.sin(adjusted_df['yaw'])
adjusted_df = adjusted_df.iloc[1:]
adjusted_df.to_csv('kalman01.csv')
