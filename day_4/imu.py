import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    # read in the data from the two files
    imu_df = pd.read_csv("./day_4/imu1.csv")  # replace filepath with yours
    vicon_df = pd.read_csv("./day_4/vi1.csv")  # replace filepath with yours
    
    # Rename columns to shorter names
    imu_df = imu_df.rename(columns={
        'Time': 'time',
        'attitude_roll_radians': 'att_x',
        'attitude_pitch_radians': 'att_y',
        'attitude_yaw_radians': 'att_z',
        'rotation_rate_x_rad_per_sec': 'rot_rate_x',
        'rotation_rate_y_rad_per_sec': 'rot_rate_y',
        'rotation_rate_z_rad_per_s': 'rot_rate_z',
        'gravity_x_G': 'grav_x',
        'gravity_y_G': 'grav_y',
        'gravity_z_G': 'grav_z',
        'user_acc_x_G': 'acc_x',
        'user_acc_y_G': 'acc_y',
        'user_acc_z_G': 'acc_z',
        'magnetic_field_x_microteslas': 'mag_x',
        'magnetic_field_y_microteslas': 'mag_y',
        'magnetic_field_z_microteslas': 'mag_z'
    })
    vicon_df = vicon_df.rename(columns={
        'Time': 'time',
        'Header': 'header',
        'translation_x': 'trans_x',
        'translation_y': 'trans_y',
        'translation_z': 'trans_z',
        'rotation_x': 'rot_quat_x',
        'rotation_y': 'rot_quat_y',
        'rotation_z': 'rot_quat_z',
        'rotation_w': 'rot_quat_w',
        'pitch': 'pitch',
        'roll': 'roll',
        'yaw': 'yaw'
    })
    
    # Convert Vicon quaternions (x, y, z, w) to Euler angles
    quaternions = vicon_df[['rot_quat_x', 'rot_quat_y', 'rot_quat_z', 'rot_quat_w']].values
    rotations = Rotation.from_quat(quaternions)
    euler_angles = rotations.as_euler('xyz', degrees=False)
    vicon_df['att_x'] = euler_angles[:, 0]
    vicon_df['att_y'] = euler_angles[:, 1]
    vicon_df['att_z'] = euler_angles[:, 2]
    
    # Plot 1: IMU attitude over time
    plt.figure(figsize=(10, 6))
    plt.plot(imu_df['time'], imu_df['att_x'], label='Roll')
    plt.plot(imu_df['time'], imu_df['att_y'], label='Pitch')
    plt.plot(imu_df['time'], imu_df['att_z'], label='Yaw')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.title('IMU Attitude Over Time')
    plt.legend()
    plt.grid()
    plt.savefig('./day_4/img/imu_attitude.png')
    plt.close()
    
    # Plot 2: Vicon attitude over time
    plt.figure(figsize=(10, 6))
    plt.plot(vicon_df['time'], vicon_df['att_x'], label='Roll')
    plt.plot(vicon_df['time'], vicon_df['att_y'], label='Pitch')
    plt.plot(vicon_df['time'], vicon_df['att_z'], label='Yaw')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.title('Vicon Attitude Over Time')
    plt.legend()
    plt.grid()
    plt.savefig('./day_4/img/vicon_attitude.png')
    plt.close()
    
    # Plot 3: IMU magnetic field over time
    plt.figure(figsize=(10, 6))
    plt.plot(imu_df['time'], imu_df['mag_x'], label='Mag X')
    plt.plot(imu_df['time'], imu_df['mag_y'], label='Mag Y')
    plt.plot(imu_df['time'], imu_df['mag_z'], label='Mag Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetic Field (µT)')
    plt.title('IMU Magnetic Field Over Time')
    plt.legend()
    plt.grid()
    plt.savefig('./day_4/img/imu_magnetic_field.png')
    plt.close()
    
    # Plot 4: IMU gravity over time
    plt.figure(figsize=(10, 6))
    plt.plot(imu_df['time'], imu_df['grav_x'], label='Gravity X')
    plt.plot(imu_df['time'], imu_df['grav_y'], label='Gravity Y')
    plt.plot(imu_df['time'], imu_df['grav_z'], label='Gravity Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('IMU Gravity Over Time')
    plt.legend()
    plt.grid()
    plt.savefig('./day_4/img/imu_gravity.png')
    plt.close()
    
    # Plot 5: IMU vs Vicon attitude over time
    plt.figure(figsize=(12, 8))
    plt.plot(imu_df['time'], imu_df['att_x'], label='IMU Roll', linestyle='-')
    plt.plot(vicon_df['time'], vicon_df['att_x'], label='Vicon Roll', linestyle='--')
    plt.plot(imu_df['time'], imu_df['att_y'], label='IMU Pitch', linestyle='-')
    plt.plot(vicon_df['time'], vicon_df['att_y'], label='Vicon Pitch', linestyle='--')
    plt.plot(imu_df['time'], imu_df['att_z'], label='IMU Yaw', linestyle='-')
    plt.plot(vicon_df['time'], vicon_df['att_z'], label='Vicon Yaw', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.title('IMU vs Vicon Attitude Comparison')
    plt.legend()
    plt.grid()
    plt.savefig('./day_4/img/imu_vs_vicon_attitude.png')
    plt.close()

    # Compute covariance and correlation matrices for IMU data
    imu_data = imu_df[['acc_x', 'acc_y', 'acc_z', 'att_x', 'att_y', 'att_z', 
                         'rot_rate_x', 'rot_rate_y', 'rot_rate_z']].values

    print("=== Acceleration (x, y, z) ===")
    acc_data = imu_df[['acc_x', 'acc_y', 'acc_z']].values
    print("Covariance:\n", np.cov(acc_data.T))
    print("Correlation:\n", np.corrcoef(acc_data.T))

    print("\n=== Attitude (x, y, z) ===")
    att_data = imu_df[['att_x', 'att_y', 'att_z']].values
    print("Covariance:\n", np.cov(att_data.T))
    print("Correlation:\n", np.corrcoef(att_data.T))

    print("\n=== Rotation Rate (x, y, z) ===")
    rot_data = imu_df[['rot_rate_x', 'rot_rate_y', 'rot_rate_z']].values
    print("Covariance:\n", np.cov(rot_data.T))
    print("Correlation:\n", np.corrcoef(rot_data.T))

    for axis in ['x', 'y', 'z']:
        print(f"\n=== Gravity vs Mag vs Acceleration ({axis}) ===")
        data = imu_df[[f'grav_{axis}', f'mag_{axis}', f'acc_{axis}']].values
        print("Covariance:\n", np.cov(data.T))
        print("Correlation:\n", np.corrcoef(data.T))

    for axis in ['x', 'y', 'z']:
        print(f"\n=== Rotation Rate vs Acceleration ({axis}) ===")
        data = imu_df[[f'rot_rate_{axis}', f'acc_{axis}']].values
        print("Covariance:\n", np.cov(data.T))
        print("Correlation:\n", np.corrcoef(data.T))

    # Covariance and CorrCoef for IMU vs Vicon
    for axis in ['x', 'y', 'z']:
        print(f"\n=== IMU vs Vicon Attitude ({axis}) ===")
        data = np.column_stack([imu_df[f'att_{axis}'].values, vicon_df[f'att_{axis}'].values])
        print("Covariance:\n", np.cov(data.T))
        print("Correlation:\n", np.corrcoef(data.T))

    # Compute residuals between IMU and Vicon attitude
    residuals_x = imu_df['att_x'] - vicon_df['att_x']
    residuals_y = imu_df['att_y'] - vicon_df['att_y']
    residuals_z = imu_df['att_z'] - vicon_df['att_z']

    # Plot residuals
    plt.figure(figsize=(12, 8))
    plt.plot(imu_df['time'], residuals_x, label='Residual Roll', linestyle='-')
    plt.plot(imu_df['time'], residuals_y, label='Residual Pitch', linestyle='-')
    plt.plot(imu_df['time'], residuals_z, label='Residual Yaw', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Residual Angle (deg)')
    plt.title('Residuals Between IMU and Vicon Attitude')
    plt.legend()
    plt.grid()
    plt.savefig('./day_4/img/residuals_attitude.png')
    plt.close()

    # Compute mean and variance of the residuals
    mean_residuals_x = np.mean(residuals_x)
    variance_residuals_x = np.var(residuals_x)
    mean_residuals_y = np.mean(residuals_y)
    variance_residuals_y = np.var(residuals_y)
    mean_residuals_z = np.mean(residuals_z)
    variance_residuals_z = np.var(residuals_z)

    print("\n=== Residuals Mean and Variance ===")
    print(f"Mean Residual Roll: {mean_residuals_x}, Variance: {variance_residuals_x}")
    print(f"Mean Residual Pitch: {mean_residuals_y}, Variance: {variance_residuals_y}")
    print(f"Mean Residual Yaw: {mean_residuals_z}, Variance: {variance_residuals_z}")

    """
    These residuals look like a lot, like they are comparable to the magnitude of the measurements to begin with.
    However, they seem relatively consistent throughout the course of the measurement.

    Part D:
    In general, I am confused as to why we attempted to correlate acceleration in different axes for example
    Perhaps high correlations would meamn lots of constrained movement that appears across axes? 
    It seems possibly expected that there are high correlations between, say, y and z rotations but not others just based on 
        the nature of the movement being captured.
    
    It seems like for the IMU vs. Vicon comparison, the correlation is quite low (less than 0.3),
        and the magnitude of the residuals is almost as high as the measurements themselves.
        But the data seem to be all over the place, so I'm having trouble even characterizing the noise
        or the quality of the measurements.

    It doesn't seem like we should trust the raw IMU data very much at all, especially not the y axis which shows near-zero
        correlation with the Vicon data.
    """