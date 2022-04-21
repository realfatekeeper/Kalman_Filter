from math import atan2
import numpy as np
import pandas as pd

# GRAVITY_VECTOR = np.array([[0,0,9.81]]).T

GRAVITY_VECTOR = np.array([[-9.81,0,0]]).T

df = pd.read_csv('ourdataset\s00\s00_raw.csv')
# print(df.columns.values.tolist())

# adjust the time
adjusted_df = df
#adjusted_df['time'] = (adjusted_df['Timestamp[nanosec]'] - adjusted_df['Timestamp[nanosec]'][0])/1000000000

odometry_df = pd.DataFrame(columns = ['time','theta_x', 'theta_y', 'theta_z','x','y','z'])

cur_angle = np.array([[0,0,0]]).T
cur_vel = np.array([[0,0,0]]).T
cur_pos = np.array([[0,0,0]]).T


for i in range(1,len(adjusted_df)):
    delta_t = adjusted_df['time'][i]-adjusted_df['time'][i-1]
    cur_omega = np.array([[
        adjusted_df['gx'][i],
        -adjusted_df['gy'][i],
        adjusted_df['gz'][i]
        ]]).T
    cur_angle = cur_angle + delta_t*cur_omega
    [_, roll, pitch] = cur_angle.T.tolist()[0]

    xm = adjusted_df['my'][i]*np.cos(roll) - adjusted_df['my'][i]*np.sin(roll)
    ym = adjusted_df['mx'][i]*np.cos(pitch) - adjusted_df['my'][i]*np.sin(roll) + adjusted_df['mz'][i]*np.cos(roll)
    yaw = atan2(ym,xm)

    [alpha, beta, gamma] = [pitch, roll, yaw]


    R = np.array([
        [np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(alpha)*np.sin(gamma)],
        [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma)],
        [             -np.sin(beta),                                           np.cos(beta)*np.sin(gamma),                                           np.cos(beta)*np.cos(gamma)]
        ])

    cur_fb = np.array([[
        adjusted_df['ax'][i],
        adjusted_df['ay'][i],
        adjusted_df['az'][i]
        ]]).T

    #Rotate to ground xyz
    cur_acc = np.matmul(R,cur_fb)

    # Remove g
    cur_acc = cur_acc - GRAVITY_VECTOR

    # Integration
    cur_vel = cur_vel + delta_t*cur_acc
    cur_pos = cur_pos + delta_t*cur_vel # + 0.5*delta_t*delta_t*cur_acc

    #print(cur_pos)
    odom = [adjusted_df['time'][i]]
    odom.extend(cur_angle.T.tolist()[0])
    odom.extend(cur_pos.T.tolist()[0])

    print(odom)
    odometry_df.loc[i] = odom

    #print("Step :%d Done",i)

odometry_df.to_csv('ourdataset\s00\s00_odom.csv')
odometry_df.to_excel('ourdataset\s00\s00_odom.xlsx')

#print(adjusted_df.to_string()) 