import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pytz

MYTZ = "Europe/Zurich"
tzinfo = pytz.timezone(MYTZ)


def add_noise(signal, noise_level=0.01):
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

def scale_signal(signal):
    scale_factor = random.uniform(0.5, 1.5)
    return signal * scale_factor

def add_drift(signal, drift_strength=0.01):
    drift = np.linspace(0, drift_strength * len(signal), len(signal))
    return signal + drift

def shift_signal(signal, shift_value=0.1):
    return signal + shift_value

def time_warp(signal, warp_factor=0.1):
    time_indices = np.arange(len(signal))
    warp_indices = np.interp(time_indices, time_indices * (1 + warp_factor), time_indices)
    return np.interp(warp_indices, time_indices, signal)

def apply_time_drift(df, drift_interval_seconds, drift_amount_seconds):
    """
    Apply time drift to the timestamps in the dataframe at regular intervals.
    
    Parameters:
    df (pd.DataFrame): DataFrame with a 'time' column.
    drift_interval_seconds (int): Interval at which to apply the drift in seconds.
    drift_amount_seconds (float): Amount of drift to apply at each interval in seconds.
    
    Returns:
    pd.DataFrame: DataFrame with drifted timestamps.
    """
    df_augmented = df.copy()
    drifted_timestamps = df_augmented['time'].copy()
    
    start_time = drifted_timestamps.iloc[0]
    
    for i in range(1, len(drifted_timestamps)):
        if (drifted_timestamps.iloc[i] - start_time).total_seconds() >= drift_interval_seconds:
            drifted_timestamps.iloc[i:] += pd.Timedelta(seconds=drift_amount_seconds)
            start_time = drifted_timestamps.iloc[i]
    
    df_augmented['time'] = drifted_timestamps
    return df_augmented


if __name__ == "__main__":
    cos_acc_csv  = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2_filtered/sensei-223/cosinuss_ear_acc_x_acc_y_acc_z/14_11_2022_cosinuss_ear_acc_x_acc_y_acc_z.csv'
    cos_temp_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2_filtered/sensei-223/cosinuss_ear_temperature/14_11_2022_cosinuss_ear_temperature.csv'
    mat_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2_filtered/sensei-223/sensomative/14_11_2022_sensomative.csv'

    # Read CSV files
    cos_acc = pd.read_csv(cos_acc_csv)
    cos_temp = pd.read_csv(cos_temp_csv)
    mat = pd.read_csv(mat_csv)

    columns = ['time', 'device1_value4', 'device1_value5', 'device1_value9']
    mat = mat[[*columns]]
    df_idx = mat[(mat['time'] >= 978.943 - 20) & (mat['time'] <= 978.943 + 2)].index
    mat = mat.loc[df_idx]

    print(mat)

    mat['timestamp'] = pd.to_datetime(mat['time'], unit='s')
    mat['timestamp'] = mat['timestamp'].dt.tz_localize('UTC').dt.tz_convert(tzinfo)

    mat_drift = apply_time_drift(mat, 10, 0.5)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(mat['timestamp'], mat['device1_value4'], label='Original', marker='.')
    plt.plot(mat_drift['timestamp'], mat_drift['device1_value4'], label='Augmented (Time Drift)', linestyle='--', marker='.')
    plt.xlabel('Time')
    plt.ylabel('device1_value4')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(mat['timestamp'], mat['device1_value5'], label='Original', marker='.')
    plt.plot(mat_drift['timestamp'], mat_drift['device1_value5'], label='Augmented (Time Drift)', linestyle='--', marker='.')
    plt.xlabel('Time')
    plt.ylabel('device1_value5')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(mat['timestamp'], mat['device1_value9'], label='Original', marker='.')
    plt.plot(mat_drift['timestamp'], mat_drift['device1_value9'], label='Augmented (Time Drift)', linestyle='--', marker='.')
    plt.xlabel('Time')
    plt.ylabel('device1_value9')
    plt.legend()

    plt.tight_layout()
    plt.show()


    # cos_acc['acc_x_noisy'] = add_noise(cos_acc['acc_x'])
    # cos_acc['acc_y_noisy'] = add_noise(cos_acc['acc_y'])
    # cos_acc['acc_z_noisy'] = add_noise(cos_acc['acc_z'])

    # cos_acc['acc_x_scaled'] = scale_signal(cos_acc['acc_x'])
    # cos_acc['acc_y_scaled'] = scale_signal(cos_acc['acc_y'])
    # cos_acc['acc_z_scaled'] = scale_signal(cos_acc['acc_z'])

    # cos_acc['acc_x_shifted'] = shift_signal(cos_acc['acc_x'])
    # cos_acc['acc_y_shifted'] = shift_signal(cos_acc['acc_y'])
    # cos_acc['acc_z_shifted'] = shift_signal(cos_acc['acc_z'])

    # cos_acc['acc_x_warped'] = time_warp(cos_acc['acc_x'])
    # cos_acc['acc_y_warped'] = time_warp(cos_acc['acc_y'])
    # cos_acc['acc_z_warped'] = time_warp(cos_acc['acc_z'])

    # plt.figure(1, figsize=(12, 8))

    # plt.subplot(2, 1, 1)
    # plt.scatter(cos_acc['time'], cos_acc['acc_x'], label='Original X')
    # plt.scatter(cos_acc['time'], cos_acc['acc_y'], label='Original Y')
    # plt.scatter(cos_acc['time'], cos_acc['acc_z'], label='Original Z')
    # plt.title('Original Data')
    # plt.xlabel('Time')
    # plt.ylabel('Acceleration')
    # plt.legend()

    # plt.subplot(2, 1, 2)
    # plt.scatter(cos_acc['time'], cos_acc['acc_x_scaled'], label='Scaled X')
    # plt.scatter(cos_acc['time'], cos_acc['acc_y_scaled'], label='Scaled Y')
    # plt.scatter(cos_acc['time'], cos_acc['acc_z_scaled'], label='Scaled Z')
    # plt.title('Augmented Data')
    # plt.xlabel('Time')
    # plt.ylabel('Acceleration')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    ######################################

    # plt.figure(2, figsize=(12, 8))

    # plt.subplot(2, 1, 1)
    # plt.scatter(cos_acc['time'], cos_acc['acc_x'], label='Original X')
    # plt.scatter(cos_acc['time'], cos_acc['acc_y'], label='Original Y')
    # plt.scatter(cos_acc['time'], cos_acc['acc_z'], label='Original Z')
    # plt.title('Original Data')
    # plt.xlabel('Time')
    # plt.ylabel('Acceleration')
    # plt.legend()

    # plt.subplot(2, 1, 2)
    # plt.scatter(cos_acc['time'], cos_acc['acc_x_shifted'], label='Shifted X')
    # plt.scatter(cos_acc['time'], cos_acc['acc_y_shifted'], label='Shifted Y')
    # plt.scatter(cos_acc['time'], cos_acc['acc_z_shifted'], label='Shifted Z')
    # plt.title('Augmented Data')
    # plt.xlabel('Time')
    # plt.ylabel('Acceleration')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    # ######################################

    # plt.figure(3, figsize=(12, 8))

    # plt.subplot(2, 1, 1)
    # plt.scatter(cos_acc['time'], cos_acc['acc_x'], label='Original X')
    # plt.scatter(cos_acc['time'], cos_acc['acc_y'], label='Original Y')
    # plt.scatter(cos_acc['time'], cos_acc['acc_z'], label='Original Z')
    # plt.title('Original Data')
    # plt.xlabel('Time')
    # plt.ylabel('Acceleration')
    # plt.legend()

    # plt.subplot(2, 1, 2)
    # plt.scatter(cos_acc['time'], cos_acc['acc_x_warped'], label='Warped X')
    # plt.scatter(cos_acc['time'], cos_acc['acc_y_warped'], label='Warped Y')
    # plt.scatter(cos_acc['time'], cos_acc['acc_z_warped'], label='Warped Z')
    # plt.title('Augmented Data')
    # plt.xlabel('Time')
    # plt.ylabel('Acceleration')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    # ######################################

    # plt.figure(4, figsize=(12, 8))

    # plt.subplot(2, 1, 1)
    # plt.scatter(cos_acc['time'], cos_acc['acc_x'], label='Original X')
    # plt.scatter(cos_acc['time'], cos_acc['acc_y'], label='Original Y')
    # plt.scatter(cos_acc['time'], cos_acc['acc_z'], label='Original Z')
    # plt.title('Original Data')
    # plt.xlabel('Time')
    # plt.ylabel('Acceleration')
    # plt.legend()

    # plt.subplot(2, 1, 2)
    # plt.scatter(cos_acc['time'], cos_acc['acc_x_noisy'], label='Noisy X')
    # plt.scatter(cos_acc['time'], cos_acc['acc_y_noisy'], label='Noisy Y')
    # plt.scatter(cos_acc['time'], cos_acc['acc_z_noisy'], label='Noisy Z')
    # plt.title('Augmented Data')
    # plt.xlabel('Time')
    # plt.ylabel('Acceleration')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()


    # # plt.scatter(cos_acc['time'], cos_acc['acc_x_noisy'], label='Noisy')
    # # plt.scatter(cos_acc['time'], cos_acc['acc_x_scaled'], label='Scaled')
    # # plt.scatter(cos_acc['time'], cos_acc['acc_x_drift'], label='Drift')
    # # plt.scatter(cos_acc['time'], cos_acc['acc_x_shifted'], label='Shifted')
    # plt.scatter(cos_acc['time'], cos_acc['acc_x_warped'], label='Warped')
    # plt.title('Augmented Accelerometer Data (X-axis)')
    # plt.xlabel('Time')
    # plt.ylabel('Acceleration')
    # plt.legend()

    # plt.subplot(3, 1, 2)
    # plt.scatter(cos_acc['time'], cos_acc['acc_y'], label='Original')
    # # plt.scatter(cos_acc['time'], cos_acc['acc_y_noisy'], label='Noisy')
    # # plt.scatter(cos_acc['time'], cos_acc['acc_y_scaled'], label='Scaled')
    # # plt.scatter(cos_acc['time'], cos_acc['acc_y_drift'], label='Drift')
    # # plt.scatter(cos_acc['time'], cos_acc['acc_y_shifted'], label='Shifted')
    # plt.scatter(cos_acc['time'], cos_acc['acc_y_warped'], label='Warped')
    # plt.title('Augmented Accelerometer Data (Y-axis)')
    # plt.xlabel('Time')
    # plt.ylabel('Acceleration')
    # plt.legend()

    # plt.subplot(3, 1, 3)
    # plt.scatter(cos_acc['time'], cos_acc['acc_z'], label='Original')
    # # plt.scatter(cos_acc['time'], cos_acc['acc_z_noisy'], label='Noisy')
    # # plt.scatter(cos_acc['time'], cos_acc['acc_z_scaled'], label='Scaled')
    # # plt.scatter(cos_acc['time'], cos_acc['acc_z_drift'], label='Drift')
    # # plt.scatter(cos_acc['time'], cos_acc['acc_z_shifted'], label='Shifted')
    # plt.scatter(cos_acc['time'], cos_acc['acc_z_warped'], label='Warped')
    # plt.title('Augmented Accelerometer Data (Z-axis)')
    # plt.xlabel('Time')
    # plt.ylabel('Acceleration')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()