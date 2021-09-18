import numpy as np
 
# State Estimation of a 2D mobile robot with EKF
# https://en.wikipedia.org/wiki/Extended_Kalman_filter#Discrete-time_predict_and_update_equations

np.set_printoptions(precision = 3, suppress = True)
 
# State matrix
A_k_minus_1 = np.array([[1.0,  0,   0],
                        [ 0, 1.0,   0],
                        [ 0,  0, 1.0 ]])
 
# Noise applied to the forward kinematics 
process_noise_v_k_minus_1 = np.array([0.01, 0.01, 0.003])
     
# State model noise covariance matrix Q_k
Q_k = np.array([[1.0,   0,   0],
                [  0, 1.0,   0],
                [  0,   0, 1.0]])
                 
# Measurement matrix H_k
H_k = np.array([[1.0,  0,   0],
                [  0, 1.0,   0],
                [  0,  0, 1.0]])
                         
# Sensor measurement noise covariance matrix R_k
R_k = np.array([[1.0,   0,    0],
                [  0, 1.0,    0],
                [  0,    0, 1.0]])  
                 
# Sensor noise
sensor_noise_w_k = np.array([0.07,0.07,0.04])
 
def compute_B(angle, d_t):
    """
    Calculates and returns the B matrix
    """
    B = np.array([[np.cos(angle)*d_t, 0],
                  [np.sin(angle)*d_t, 0],
                  [0, d_t]])
    return B
 
def ekf(z_k_observation_vector, state_estimate_k_minus_1, 
        control_vector_k_minus_1, P_k_minus_1, dk):
    """
    Extended Kalman Filter.
    """

    ## Predict the state estimate
    ## X_t | t_minus_1 = A_x_minus_1 @ X_t_minus_1 + B_x_minus_1 @ U_t_minus_1 + V_t_minus_1
    state_estimate_k = A_k_minus_1 @ (
            state_estimate_k_minus_1) + (
            compute_B(state_estimate_k_minus_1[2],dk)) @ (
            control_vector_k_minus_1) + (
            process_noise_v_k_minus_1)
             
    print(f'State Estimate = {state_estimate_k}')
             
    # Predict the state covariance estimate
    # Sigma_t_x | t_minus_1 = (A_x_minus_1 @ Sigma_t_minus_1 @ A_x_minus_1.T) + Q_t
    P_k = A_k_minus_1 @ P_k_minus_1 @ A_k_minus_1.T + (Q_k)
         
    ## Update
    X_z = (H_k @ state_estimate_k) + (sensor_noise_w_k)
 
    print(f'Sensor Observation = {z_k_observation_vector}')
             
    # Calculate the measurement residual covariance
    # Sigma_t_u | t_minus_1 = (B_x_minus_1 @ Sigma_t_x @ B_x_minus_1.T) + R_t
    S_k = H_k @ P_k @ H_k.T + R_k
         
    # Calculate the near-optimal Kalman gain
    # K_t 
    K_k = P_k @ H_k.T @ np.linalg.pinv(S_k)
         
    # Calculate an updated state estimate for time k
    # X_t | t = X_t | t_minus_1 + K_t @ (Z_t - X_z)
    state_estimate_k = state_estimate_k + (K_k @ (z_k_observation_vector - X_z))
     
    # Update the state covariance estimate for time k
    # Sigma_t_x
    P_k = P_k - (K_k @ H_k @ P_k)
     
    # Print the best (near-optimal) estimate of the current state of the robot
    print(f'State Estimate Corrected = {state_estimate_k}')
 
    # Return the updated state and covariance estimates
    return state_estimate_k, P_k
     
def main():
 
    # We start at time k=1
    k = 1
     
    # Time interval in seconds
    dk = 1
 
    # Create a list of sensor observations at successive timesteps
    z_k = np.array([[4.721, 0.143,0.006], # k=1
                    [9.353, 0.284,0.007], # k=2
                    [14.773, 0.422,0.009],# k=3
                    [18.246, 0.555,0.011], # k=4
                    [22.609, 0.715,0.012]])# k=5
                     
    # The estimated state vector at time k-1
    state_estimate_k_minus_1 = np.array([0.0, 0.0, 0.0])
     
    # The control input vector at time k-1
    control_vector_k_minus_1 = np.array([3.0, 0.0])
     
    # State covariance matrix P_k_minus_1
    P_k_minus_1 = np.array([[0.1,  0,   0],
                            [  0, 0.1,   0],
                            [  0,  0, 0.1]])
                             
    # Start at k=1 and go through each of the 5 sensor observations
    for k, obs_vector_z_k in enumerate(z_k,start=1):
     
        # Print the current timestep
        print(f'Timestep k = {k}')  
         
        # Run the Extended Kalman Filter 
        optimal_state_estimate_k, covariance_estimate_k = ekf(
            obs_vector_z_k,
            state_estimate_k_minus_1,
            control_vector_k_minus_1,
            P_k_minus_1,
            dk)
         
        # Update the variable values
        state_estimate_k_minus_1 = optimal_state_estimate_k
        P_k_minus_1 = covariance_estimate_k
         
        # Print a blank line
        print()
 
# Program starts running here with the main method  
if __name__ == '__main__':
    main()