import numpy as np
import pandas as pd
# Author: Akarawut Krinukul
# Description: Extended Kalman Filter for imu GPS fusion adapted from example on https://automaticaddison.com (two-wheeled mobile robot)

# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=3, suppress=True)

# A matrix
# 3x3 matrix -> number of states x number of states matrix
# Expresses how the state of the system [x,y,yaw,vx,vx] changes
# from k-1 to k when no control command is executed.
# Typically a robot on wheels only drives when the wheels are told to turn.
# For this case, A is the identity matrix.
# A is sometimes F in the literature.
A_k_minus_1 = np.identity(5)

# Noise applied to the forward kinematics (calculation
# of the estimated state at time k from the state
# transition model of the mobile robot). This is a vector
# with the number of elements equal to the number of states
process_noise_v_k_minus_1 = np.array([0.01, 0.01, 0.003, 0.01, 0.01])

# State model noise covariance matrix Q_k
# When Q is large, the Kalman Filter tracks large changes in
# the sensor measurements more closely than for smaller Q.
# Q is a square matrix that has the same number of rows as states.
Q_k = np.identity(5)

# Measurement matrix H_k
# Used to convert the predicted state estimate at time k
# into predicted sensor measurements at time k.
# In this case, H will be the identity matrix since the
# estimated state maps directly to state measurements from the
# odometry data [GPSx, GPSy]
# H has the same number of rows as sensor measurements
# and same number of columns as states.
H_k = np.array([[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0]])

# Sensor measurement noise covariance matrix R_k
# Has the same number of rows and columns as sensor measurements.
# If we are sure about the measurements, R will be near zero.
R_k = np.identity(3) * 0.01

# Sensor noise. This is a vector with the
# number of elements equal to the number of sensor measurements.
sensor_noise_w_k = np.array([0.07, 0.07])


def getA(deltak):
    """
    Calculates and returns the A matrix
    5x5 matix -> number of states x number of states 
    The control inputs are the forward speed and the
    rotation rate around the z axis from the x-axis in the 
    counterclockwise direction.
    Expresses how the state of the system [x,y,yaw,vx,vy] changes
    from k-1 to k due to the system itself.
    :param deltak: The change in time from time step k-1 to k in sec
    """
    A = np.array([[1, 0, 0, deltak, 0],
                  [0, 1, 0, 0, deltak],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])
    return A


def getB(yaw, deltak):
    """
    Calculates and returns the B matrix
    5x2 matix -> number of states x number of control inputs
    The control inputs are the forward speed and the
    rotation rate around the z axis from the x-axis in the 
    counterclockwise direction.
    [a,yaw_rate]
    Expresses how the state of the system [x,y,yaw,vx,vy] changes
    from k-1 to k due to the control commands (i.e. control input).
    :param yaw_rate: The yaw rate (rotation angle rate around the z axis) in rad/s
    :param deltak: The change in time from time step k-1 to k in sec
    """
    B = np.array([[0, 0],
                  [0, 0],
                  [0, deltak],
                  [np.cos(yaw)*deltak, 0],
                  [np.sin(yaw)*deltak, 0]])
    return B


def getHx(yaw, lastyaw, vx, vy):
    dist = np.sqrt((pow(vx, 2)+pow(vy, 2)))
    Hx = np.array([yaw-lastyaw, vx/dist, vy/dist])
    return Hx


def get_H_k(vx, vy):
    dist = pow(np.sqrt((pow(vx, 2)+pow(vy, 2))), 3)
    H_k = np.array([[0, 0, 1, 0, 0],
                    [0, 0, 0, vy**2/dist, -vx*vy/dist],
                    [0, 0, 0,  -vx*vy/dist, vx**2/dist]])
    return H_k


def ekf_predict(state_estimate_k_minus_1,
                control_vector_k_minus_1, P_k_minus_1, dk):
    """
    Extended Kalman Filter. Fuses noisy sensor measurement to 
    create an optimal estimate of the state of the robotic system.

    INPUT
        :param z_k_observation_vector The observation from the Odometry
            5x1 NumPy Array [x,y,yaw,vx,vy] in the global reference frame
            in [meters,meters,radians,m/s,m/s].
        :param state_estimate_k_minus_1 The state estimate at time k-1
            5x1 NumPy Array [x,y,yaw,vx,vy] in the global reference frame
            in [meters,meters,radians].
        :param control_vector_k_minus_1 The control vector applied at time k-1
            2x1 NumPy Array [a,yaw rate] in the global reference frame
            in [meters per second,meters per second,radians per second].
        :param P_k_minus_1 The state covariance matrix estimate at time k-1
            5x5 NumPy Array
        :param dk Time interval in seconds

    OUTPUT
        :return state_estimate_k near-optimal state estimate at time k  
            5x1 NumPy Array ---> [meters,meters,radians]
        :return P_k state covariance_estimate for time k
            5x5 NumPy Array                 
    """
    ######################### Predict #############################
    # Predict the state estimate at time k based on the state
    # estimate at time k-1 and the control input applied at time k-1.
    state_estimate_k = getA(dk) @ (state_estimate_k_minus_1) + (
        getB(state_estimate_k_minus_1[2], dk)) @ (
        control_vector_k_minus_1) + (
        process_noise_v_k_minus_1)

    print(f'State Estimate Before EKF={state_estimate_k}')

    # Predict the state covariance estimate based on the previous
    # covariance and some noise
    P_k = A_k_minus_1 @ P_k_minus_1 @ A_k_minus_1.T + (
        Q_k)

    # Return the updated state and covariance estimates
    return state_estimate_k, P_k
 

def ekf_update(z_k_observation_vector, state_estimate_k, state_estimate_k_minus_1 , P_k):
    """
    Low DR process
    Update the EKF based on updated measurement

    INPUT
        :param z_k_observation_vector The observation from the Odometry
            5x1 NumPy Array [x,y,yaw,vx,vy] in the global reference frame
            in [meters,meters,radians,m/s,m/s].


    OUTPUT
        :return state_estimate_k near-optimal state estimate at time k  
            5x1 NumPy Array ---> [meters,meters,radians]
        :return P_k state covariance_estimate for time k
            5x5 NumPy Array                 
    """
    ################### Update (Correct) ##########################
    # Calculate the difference between the actual sensor measurements
    # at time k minus what the measurement model predicted
    # the sensor measurements would be for the current timestep k.
    measurement_residual_y_k = z_k_observation_vector - \
        getHx(state_estimate_k[2],state_estimate_k_minus_1[2], state_estimate_k[3], state_estimate_k[4])

    print(f'Observation={z_k_observation_vector}')

    H_k = get_H_k(state_estimate_k[3], state_estimate_k[4])

    # Calculate the measurement residual covariance
    S_k = H_k @ P_k @ H_k.T + R_k

    # Calculate the near-optimal Kalman gain
    # We use pseudoinverse since some of the matrices might be
    # non-square or singular.
    K_k = P_k @ H_k.T @ np.linalg.pinv(S_k)

    # Calculate an updated state estimate for time k
    state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)

    # Update the state covariance estimate for time k
    P_k = P_k - (K_k @ H_k @ P_k)

    # Print the best (near-optimal) estimate of the current state of the robot
    print(f'State Estimate After EKF Update ={state_estimate_k}')

    return state_estimate_k, P_k


def ekf(z_k_observation_vector, state_estimate_k_minus_1,
        control_vector_k_minus_1, P_k_minus_1, dk, update=True):
    """
    Extended Kalman Filter. Fuses noisy sensor measurement to 
    create an optimal estimate of the state of the robotic system.

    INPUT
        :param z_k_observation_vector The observation from the Odometry
            5x1 NumPy Array [x,y,yaw,vx,vy] in the global reference frame
            in [meters,meters,radians,m/s,m/s].
        :param state_estimate_k_minus_1 The state estimate at time k-1
            5x1 NumPy Array [x,y,yaw,vx,vy] in the global reference frame
            in [meters,meters,radians].
        :param control_vector_k_minus_1 The control vector applied at time k-1
            2x1 NumPy Array [a,yaw rate] in the global reference frame
            in [meters per second,meters per second,radians per second].
        :param P_k_minus_1 The state covariance matrix estimate at time k-1
            5x5 NumPy Array
        :param dk Time interval in seconds

    OUTPUT
        :return state_estimate_k near-optimal state estimate at time k  
            5x1 NumPy Array ---> [meters,meters,radians]
        :return P_k state covariance_estimate for time k
            5x5 NumPy Array                 
    """
    state_estimate_k, P_k = ekf_predict(state_estimate_k_minus_1,
                                        control_vector_k_minus_1, P_k_minus_1, dk)
    if update:
        state_estimate_k, P_k = ekf_update(
            z_k_observation_vector*0.1, state_estimate_k,state_estimate_k_minus_1, P_k)

    return state_estimate_k, P_k


def main():
    # Read preprocessed data from csv
    data = pd.read_csv('kalman00.csv')
    VOdf = pd.read_csv('VO00.csv')
    # We start at time k=1
    k = 1

    # Time interval in seconds
    dk = 0.1

    ## Change step of VO to read
    VOstep = 1

    # The estimated state vector at time k-1 in the global reference frame.
    # [x_k_minus_1, y_k_minus_1, yaw_k_minus_1, v_x_k_minus_1 , v_y_k_minus_1]
    # [meters, meters, radians]
    state_estimate_k_minus_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # The control input vector at time k-1 in the global reference frame.
    # [v, yaw_rate]
    # [meters/second, radians/second]
    # In the literature, this is commonly u.
    # Because there is no angular velocity and the robot begins at the
    # origin with a 0 radians yaw angle, this robot is traveling along
    # the positive x-axis in the global reference frame.

    # State covariance matrix P_k_minus_1
    # This matrix has the same number of rows (and columns) as the
    # number of states (i.e. 5x5 matrix). P is sometimes referred
    # to as Sigma in the literature. It represents an estimate of
    # the accuracy of the state estimate at time k made using the
    # state transition matrix. We start off with guessed values.
    P_k_minus_1 = np.identity(5)*0.1
    track = []
    history = []
    # Start from the first row and go through each of the sensor observations,
    # one at a time.
    for k, row in data.iterrows():
        if k == 0:
            # Assume theta = 0 at first timestep
            np.array([0.0, 0.0, row["omega"]*dk, 0.0, 0.0])

            """
            track.append([state_estimate_k_minus_1[0],
            state_estimate_k_minus_1[1], row["GPSx"], row["GPSy"]])
            """

            history.append([state_estimate_k_minus_1[i] for i in range(5)])
            continue
        # obs_vector_z_k is the data observed from external sensors
        Vrow = VOdf.iloc[k*VOstep]
        obs_vector_z_k = np.array([Vrow["yawVO"],Vrow["vxVO"], Vrow["vyVO"]])

        control_vector_k_minus_1 = np.array([row["a"], row["omega"]])
        # Print the current timestep
        print(f'Timestep k={k}')

        """
        uncomment for low DR update
        if row["GPSx"] != track[-1][-2] and row["GPSy"] != track[-1][-1]:
            update = True
        else:
            update = False
        
        """

        # Run the Extended Kalman Filter and store the
        # near-optimal state and covariance estimates

        optimal_state_estimate_k, covariance_estimate_k = ekf(
            obs_vector_z_k,  # Most recent sensor measurement
            state_estimate_k_minus_1,  # Our most recent estimate of the state
            control_vector_k_minus_1,  # Our most recent control input
            P_k_minus_1,  # Our most recent state covariance matrix
            dk)  # Time interval

        # Get ready for the next timestep by updating the variable values
        state_estimate_k_minus_1 = optimal_state_estimate_k
        P_k_minus_1 = covariance_estimate_k

        
        track.append([state_estimate_k_minus_1[0],
        state_estimate_k_minus_1[1], row["GPSx"], row["GPSy"]])
        
        history.append([state_estimate_k_minus_1[i] for i in range(5)])

        # Print a blank line
        print()

    track = np.array(track)
    history = np.array(history)

    print('Saving Track record and History')

    
    result = pd.DataFrame(track, columns=['x', 'y', 'GPSx', 'GPSy'])
    result.to_csv('result00.csv')
    

    histories = pd.DataFrame(history, columns=['x', 'y', 'yaw', 'vx', 'vy'])
    histories.to_csv('history00.csv')


# Program starts running here with the main method
if __name__ == '__main__':
    main()
