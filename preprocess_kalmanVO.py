import numpy as np
import math
import pandas as pd
import pyproj

# Use with kalman02 and attempts
'''
# some constant values for utm.
f = 1/298.257223563
n = f / (2 - f)
alpha1 = 0.5*n - (2/3)*n**2 + (5/16)*n**3
alpha2 = (13/48)*n**2 - (3/5)*n**3
alpha3 = (61/240)*n**3
beta1  = 0.5*n - (2/3)*n**2 + (37/96)*n**3
beta2  = (1/48)*n**2 - (1/15)*n**3
beta3  = (17/480)*n**3
sigma1 = 2*n - (2/3)*n**2 - 2*n**3
sigma2 = (7/3)*n**2 - (8/5)*n**3
sigma3 = (56/15)*n**3

_t = 2*np.sqrt(n)/(1+n) # constant for t:tanh

def utm_intermediate(lat, long):
    """
    Compute intermediate values for utm coordinates
    n = f / (2-f)
    """

    t = np.sinh(np.arctanh(np.sin(lat)) - _t*np.arctanh(_t*np.sin(lat)) )

    xi_prime = np.arctanh(t / np.cos(long - ))
'''
# From https://gist.github.com/twpayne/4409500 

_projections = {}
# coordinates = (lng, lat)

def zone(coordinates):
    if 56 <= coordinates[1] < 64 and 3 <= coordinates[0] < 12:
        return 32
    if 72 <= coordinates[1] < 84 and 0 <= coordinates[0] < 42:
        if coordinates[0] < 9:
            return 31
        elif coordinates[0] < 21:
            return 33
        elif coordinates[0] < 33:
            return 35
        return 37
    return int((coordinates[0] + 180) / 6) + 1


def letter(coordinates):
    return 'CDEFGHJKLMNPQRSTUVWXX'[int((coordinates[1] + 80) / 8)]


def project(coordinates):
    z = zone(coordinates)
    l = letter(coordinates)
    if z not in _projections:
        _projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
    x, y = _projections[z](coordinates[0], coordinates[1])
    if y < 0:
        y += 10000000
    return z, l, x, y


def unproject(z, l, x, y):
    if z not in _projections:
        _projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
    if l < 'N':
        y -= 10000000
    lng, lat = _projections[z](x, y, inverse=True)
    return (lng, lat)



df = pd.read_csv(r'raw_data\00\00.csv')
df
# print(df.columns.values.tolist())
# ROT_BIAS = [0,0,0]
# TRANS_BIAS = [0,0,0]
# adjust the time
adjusted_df = pd.DataFrame()
adjusted_df['time'] = (df['timestamp'] - df['timestamp'][0])/1000
# df["GPS.lat"]
# Calculate location from GPS
xs, ys = df["GPS.lat"],df["GPS.long"]

def sphere_proj(xs, ys):
    """ ROT_BIAS = [-3.7034333488898317e-06, -6.277694338380008e-09, 4.123992894451507e-06]
    TRANS_BIAS = [0.03137413793138769, 0.006497261588525887, 0.03793959105307] """

    EARTH_RADIUS = 6371000
    PI = 3.14159265

    phi = np.radians(xs) # phi in formula
    ld = np.radians(ys) # lambda in formula
    x = EARTH_RADIUS * ld
    y = EARTH_RADIUS * np.log(np.tan(PI/4 +phi/2))
    return x, y


# x0,y0 = sphere_proj(xs, ys)

#UTM projection.

proj_xy = np.array([project((lng, lat)) for lng, lat in zip(ys, xs)])
x = proj_xy[:, 2].astype(np.float64)
y = proj_xy[:, 3].astype(np.float64)
"""
# import matplotlib.pyplot as plt
# set to 0 for both sets
# x0 = x0 - x0[0]
# y0 = y0 - y0[0]
# x = x - x[0]
# y = y - y[0]
# plt.scatter(x0, y0,s = 0.1,c='r', label='Sphere Projection')
# plt.scatter(x, y,s = 0.1,c='b', label='UTM WGS84')
# plt.legend()
# plt.axis('equal')
# plt.show()

# import sys
# sys.exit(0)
"""

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])






a = np.sqrt((df["icm4x6xx_accelerometer.x"]+9.81)**2 + df["icm4x6xx_accelerometer.y"]**2+df["icm4x6xx_accelerometer.z"]**2)
adjusted_df['GPSx'] = x
adjusted_df['GPSy'] = y
adjusted_df['a'] = a
adjusted_df['omega'] = -1*df["icm4x6xx_gyroscope.x"]
adjusted_df['yaw'] = -1* np.radians(df["orientation.x"])

adjusted_df['GPSx'] = adjusted_df['GPSx'] - adjusted_df.iloc[0].at['GPSx']
adjusted_df['GPSy'] = adjusted_df['GPSy'] - adjusted_df.iloc[0].at['GPSy']

adjusted_df = adjusted_df[adjusted_df['GPSx'] != 0]
adjusted_df['GPSx'] = adjusted_df['GPSx'] - adjusted_df.iloc[0].at['GPSx']
adjusted_df['GPSy'] = adjusted_df['GPSy'] - adjusted_df.iloc[0].at['GPSy']

adjusted_df.to_csv('kalman00.csv')


VO_data = pd.read_csv(r"raw_data\00\OTR_00_transformation.txt", sep=" ", header=None)
VO_df = pd.DataFrame()
VO_df['vxVO'] = np.where(np.sqrt(VO_data[3] * VO_data[3] +  VO_data[11] *VO_data[11]) == 0,0,VO_data[3]/(VO_data[3] * VO_data[3] +  VO_data[11] *VO_data[11]))
VO_df['vyVO'] = np.where(np.sqrt(VO_data[3] * VO_data[3] +  VO_data[11] *VO_data[11]) == 0,0,VO_data[11]/(VO_data[3] * VO_data[3] +  VO_data[11] *VO_data[11]))
VO_df['yawVO'] = np.where(np.sqrt(VO_data[0] * VO_data[0] +  VO_data[4] *VO_data[4]) == 0,0,np.arctan2(-VO_data[7],VO_data[0] * VO_data[0] +  VO_data[4] *VO_data[4]))
VO_df.to_csv('VO00.csv')




