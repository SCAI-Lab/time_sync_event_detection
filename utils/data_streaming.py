"""
An exmaple file that simulate the data streaming using threading
"""

import pandas as pd
import time
import threading
import pytz

MYTZ = "Europe/Zurich"
tzinfo = pytz.timezone(MYTZ)


def stream_data(sensor_data, sampling_interval, init_delay, sensor_name):
    time.sleep(init_delay)
    for _, row in sensor_data.iterrows():
        print(f"{sensor_name} data: {row[:3]} at {row['time']}")
        time.sleep(sampling_interval)


if __name__ == "__main__":
    mat1_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/sensomative/user_223_2022-11-14_15-32-35-310[1].csv'
    mat2_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/sensomative/user_223b_2022-11-14_16-39-49-858[1].csv'
    cos_acc_csv  = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/cosinuss_ear_acc_x_acc_y_acc_z/K41C.9ZA0_2022-11-07_10-44-13_acc_x_acc_y_acc_z.csv'
    
    # Read CSV files
    mat1 = pd.read_csv(mat1_csv)
    mat2 = pd.read_csv(mat2_csv)
    mat = pd.concat([mat1, mat2], axis=0, ignore_index=True)
    mat = mat.sort_values(by='time')
    cos_acc = pd.read_csv(cos_acc_csv)

    df_idx = mat[(mat['time'] >= 1668438354 - 20) & (mat['time'] <= 1668438354 + 2)].index
    mat = mat.loc[df_idx]

    df_idx = cos_acc[(cos_acc['time'] >= 1668438354 - 20) & (cos_acc['time'] <= 1668438354 + 2)].index
    cos_acc = cos_acc.loc[df_idx]

    cos_acc['timestamp'] = cos_acc['time']
    mat['timestamp'] = mat['time']

    cos_acc['time'] = pd.to_datetime(cos_acc['time'], unit='s')
    mat['time'] = pd.to_datetime(mat['time'], unit='s')

    mat['time'] = mat['time'].dt.tz_localize('UTC').dt.tz_convert(tzinfo)
    cos_acc['time'] = cos_acc['time'].dt.tz_localize('UTC').dt.tz_convert(tzinfo)


    # Calculate the sampling intervals (in seconds)
    cos_acc_sampling_interval = (cos_acc['time'].iloc[-1] - cos_acc['time'].iloc[0]).total_seconds() / len(cos_acc['time'])
    mat_sampling_interval = (mat['time'].iloc[-1] - mat['time'].iloc[0]).total_seconds() / len(mat['time'])

    start_times = [cos_acc['time'].min(), mat['time'].min()]
    earliest_start_time = min(start_times)
    print(f"start time: {start_times}")


    cos_acc_delay = (cos_acc['time'].min() - earliest_start_time).total_seconds()
    mat_delay = (mat['time'].min() - earliest_start_time).total_seconds()

    print(f"cos_acc_sampling_interval: {cos_acc_sampling_interval}")
    print(f"mat_sampling_interval: {mat_sampling_interval}")
        
    print(f"cos_acc_delay: {cos_acc_delay}")
    print(f"mat_delay: {mat_delay}")

    # Create threads for each sensor
    cos_acc_thread = threading.Thread(target=stream_data, args=(cos_acc, cos_acc_sampling_interval, cos_acc_delay, 'ACC'))
    mat_thread = threading.Thread(target=stream_data, args=(mat, mat_sampling_interval, mat_delay, 'Pressure'))

    # Start the threads
    cos_acc_thread.start()
    mat_thread.start()


