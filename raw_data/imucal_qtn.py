import numpy as np
import pandas as pd
import math
# GRAVITY_VECTOR = np.array([[0,0,9.81]]).T

GRAVITY_VECTOR = np.array([[-9.81,0,0]]).T

# GLOBALLY DECLARED, required for Mahony filter
# vector to hold quaternion

q = [1,0,0,0]


def quarternion_update(gx,gy,gz,delta_t):
    # integrate rate of change of quarternion
    delta_t = 0.5 * delta_t
    gx = gx*delta_t
    gy = gy*delta_t
    gz = gz*delta_t

    # calculate quarternion
    qa = q[0]
    qb = q[1]
    qc = q[2]
    q[0] += (-qb * gx - qc * gy - q[3] * gz)
    q[1] += (qa * gx + qc * gz - q[3] * gy)
    q[2] += (qa * gy - qb * gz + q[3] * gx)
    q[3] += (qa * gz + qb * gy - qc * gx)

    # renormalized quarternion

    recipNorm = 1.0 / math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    q[0] = q[0] * recipNorm
    q[1] = q[1] * recipNorm
    q[2] = q[2] * recipNorm
    q[3] = q[3] * recipNorm


def main():
    df = pd.read_csv('ourdataset\\06\d06_calib.csv')
    # print(df.columns.values.tolist())

    # adjust the time
    adjusted_df = df
    #adjusted_df['time'] = (adjusted_df['Timestamp[nanosec]'] - adjusted_df['Timestamp[nanosec]'][0])/1000000000

    odometry_df = pd.DataFrame(columns = ['time','theta_x', 'theta_y', 'theta_z','x','y','z'])

    cur_angle = np.array([[0,0,0]]).T
    cur_vel = np.array([[0,0,0]]).T
    cur_pos = np.array([[0,0,0]]).T

    #align sensor data to Tait-Bryan systems

    for i in range(1,len(adjusted_df)):
        delta_t = adjusted_df['time'][i]-adjusted_df['time'][i-1]
        [gx, gy, gz] = [
            adjusted_df['gx'][i],
            adjusted_df['gy'][i],
            adjusted_df['gz'][i]
            ]

        quarternion_update(gx,gy,gz,delta_t)

        gamma = math.atan2((q[0] * q[1] + q[2] * q[3]), 0.5 - (q[1] * q[1] + q[2] * q[2])) # roll
        beta = math.asin(2.0 * (q[0] * q[2] - q[1] * q[3])) # pitch
        alpha = -math.atan2((q[1] * q[2] + q[0] * q[3]), 0.5 - (q[2] * q[2] + q[3] * q[3])) # yaw

        cur_angle = np.array([[
                                gamma,
                                beta,
                                alpha
                                ]]).T

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
        cur_pos = cur_pos + delta_t*cur_vel + 0.5*delta_t*delta_t*cur_acc

        #print(cur_pos)
        odom = [adjusted_df['time'][i]]
        odom.extend(cur_angle.T.tolist()[0])
        odom.extend(cur_pos.T.tolist()[0])

        print(odom)
        odometry_df.loc[i] = odom

    odometry_df.to_csv('ourdataset\\06\d06_odom_qtn.csv')
    odometry_df.to_excel('ourdataset\\06\d06_odom_qtn.xlsx')

#print(adjusted_df.to_string()) 

if __name__ == "__main__":
    main()