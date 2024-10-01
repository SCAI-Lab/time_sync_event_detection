import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import pandas as pd

import pytz

MYTZ = "Europe/Zurich"
tzinfo = pytz.timezone(MYTZ)


def synchronization(
        ref_name : str, 
        dfs : dict, 
        prev_frame : int, 
        frame : int,
        cur_sensor_name=None):
    
    ref = dfs[ref_name]
    ref_frame = ref.iloc[prev_frame : frame + 1]

    if ref_frame.empty:
        print("Reference frame is empty. Skipping synchronization.")
        return dfs, prev_frame

    ref_start_time = ref_frame['time'].iloc[0]
    ref_end_time = ref_frame['time'].iloc[-1]

    ref_start_timestamp = ref_start_time.timestamp()
    ref_end_timestamp = ref_end_time.timestamp()

    for i, (sensor_name, df) in enumerate(dfs.items()):
        if sensor_name == ref_name: continue
        # print(f"beginning Dtype of corrected_time: {df['time'].dtype}")
        df_frame = df.iloc[prev_frame : frame + 1]
        if df_frame.empty:
            print(f"DataFrame for {sensor_name} is empty. Skipping synchronization for this sensor.")
            continue

        df_start_time = df_frame['time'].iloc[0]
        df_end_time = df_frame['time'].iloc[-1]

       
        df_start_timestamp = df_start_time.timestamp()
        df_end_timestamp = df_end_time.timestamp()

        # normalize the shifted timestamp to 0~1        
        # df_normalized_times = (df_frame['time'].apply(lambda x: x.timestamp()) - df_start_timestamp) / (df_end_timestamp - df_start_timestamp)
        
        df_normalized_times = (
            df_frame['time'].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan) - df_start_timestamp
        ) / (df_end_timestamp - df_start_timestamp)

        # Drop NaN values
        df_normalized_times = df_normalized_times.dropna()

        if len(df_normalized_times) == 0:
            print(f"DataFrame for {sensor_name} after normalization is empty. Skipping synchronization for this sensor.")
            continue

        aligned_timestamps = ref_start_timestamp + df_normalized_times * (ref_end_timestamp - ref_start_timestamp)

        aligned_times = pd.to_datetime(aligned_timestamps, unit='s')
        aligned_times = aligned_times.dt.tz_localize('UTC').dt.tz_convert(tzinfo)

        df.loc[prev_frame : frame, 'time'] = aligned_times
        df.loc[prev_frame : frame, 'timestamp'] = aligned_timestamps
        
    prev_frame = frame
    return dfs, prev_frame