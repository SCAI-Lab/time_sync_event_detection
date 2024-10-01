import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))

import pandas as pd
from matplotlib.animation import FuncAnimation
import pytz
import time
import threading
import yaml

from evaluation.evaluation import *
from utils.data_augmentation_utils import *
from src.DataReceiver import DataReceiver

# setup timezone for timestamp
MYTZ = "Europe/Zurich"
tzinfo = pytz.timezone(MYTZ)


def stream_data(sensor_data: pd.DataFrame, sampling_interval: float, init_delay: float, sensor_name: str, live_dr: DataReceiver, completion_event: threading.Event):
    time.sleep(init_delay)
    for _, row in sensor_data.iterrows():
        row = row.to_frame()
        live_dr.update_data(sensor_name, row)
        live_dr.event_detect(sensor_name)
        time.sleep(sampling_interval)
    completion_event.set()
    print(f"All rows have been read for sensor {sensor_name}")
    if all(event.is_set() for event in completion_events.values()):
        live_dr.all_data_streams_completed.set()  # Signal to DataReceiver

    
def start_plotting(live_dr: DataReceiver):
    live_dr.start_plotting()

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    config = load_config('config/csv_config.yaml')

    sensors_config = {sensor['name']: sensor for sensor in config['sensors']}
    plotting = config['plotting']

    mat1_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/sensomative/user_223_2022-11-14_15-32-35-310[1].csv'
    mat2_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/sensomative/user_223b_2022-11-14_16-39-49-858[1].csv'
    cos_acc_csv  = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/cosinuss_ear_acc_x_acc_y_acc_z/K41C.9ZA0_2022-11-07_10-44-13_acc_x_acc_y_acc_z.csv'
    viva1_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/vivalnk_vv330_acceleration/20221114/20221114_1400.csv'
    viva2_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/vivalnk_vv330_acceleration/20221114/20221114_1500.csv'
    cor_acc_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/corsano_wrist_acc/2022-11-14.csv'


    # Read CSV files
    mat1 = pd.read_csv(mat1_csv)
    mat2 = pd.read_csv(mat2_csv)
    mat = pd.concat([mat1, mat2], axis=0, ignore_index=True)
    mat = mat.sort_values(by='time')
    viva1 = pd.read_csv(viva1_csv)
    viva2 = pd.read_csv(viva2_csv)
    viva_acc = pd.concat([viva1, viva2], axis=0, ignore_index=True)
    viva_acc = viva_acc.sort_values(by='time')
    cos_acc = pd.read_csv(cos_acc_csv)
    cor_acc = pd.read_csv(cor_acc_csv)

    # selected time
    df_idx = mat[(mat['time'] >= 1668441595 - 50) & (mat['time'] <= 1668441595)].index
    mat = mat.loc[df_idx]

    df_idx = cos_acc[(cos_acc['time'] >= 1668441595 - 50) & (cos_acc['time'] <= 1668441595)].index
    cos_acc = cos_acc.loc[df_idx]

    df_idx = viva_acc[(viva_acc['time'] >= 1668441595 - 50) & (viva_acc['time'] <= 1668441595)].index
    viva_acc = viva_acc.loc[df_idx]

    df_idx = cor_acc[(cor_acc['time'] >= 1668441595 - 50) & (cor_acc['time'] <= 1668441595)].index
    cor_acc = cor_acc.loc[df_idx]

    cos_acc['timestamp'] = cos_acc['time']
    cor_acc['timestamp'] = cor_acc['time']
    mat['timestamp'] = mat['time']
    viva_acc['timestamp'] = viva_acc['time']

    cos_acc['time'] = pd.to_datetime(cos_acc['time'], unit='s')
    cor_acc['time'] = pd.to_datetime(cor_acc['time'], unit='s')
    mat['time'] = pd.to_datetime(mat['time'], unit='s')
    viva_acc['time'] = pd.to_datetime(viva_acc['time'], unit='s')

    mat['time'] = mat['time'].dt.tz_localize('UTC').dt.tz_convert(tzinfo)
    cos_acc['time'] = cos_acc['time'].dt.tz_localize('UTC').dt.tz_convert(tzinfo)
    cor_acc['time'] = cor_acc['time'].dt.tz_localize('UTC').dt.tz_convert(tzinfo)
    viva_acc['time'] = viva_acc['time'].dt.tz_localize('UTC').dt.tz_convert(tzinfo)

    cos_acc['time'] = cos_acc['time'] + pd.Timedelta(seconds=3) # add 3 seconds to compensate drift of cos
    cor_acc['time'] = cor_acc['time'] - pd.Timedelta(seconds=0.5)

    df_map = {'mat': mat,
              'cos': cos_acc,
              'cor': cor_acc,
              'viva_acc': viva_acc}
    
    # plt.figure(figsize=(12, 6))
    # plt.plot(viva_acc['time'], viva_acc['x'], label='acc_x', marker='.')
    # plt.plot(viva_acc['time'], viva_acc['y'], label='acc_y', marker='.')
    # plt.plot(viva_acc['time'], viva_acc['z'], label='acc_z', marker='.')
    
    
    live_dr = DataReceiver(sensors_config=sensors_config, plotting=plotting, df_map=df_map)
    completion_events = {sensors_config[sensor]['name']: threading.Event() for sensor in sensors_config}
    
    threads = []
    for sensor in sensors_config:
        name = sensors_config[sensor]['name']
        df = df_map[name]
        sampling_interval = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds() / len(df['time'])
        delay = (df['time'].min() - min(df['time'])).total_seconds()
        # if name == 'cos':
        #     df = apply_time_drift(df, 10, 0.5)
        thread = threading.Thread(target=stream_data, args=(df, sampling_interval, delay, name, live_dr, completion_events[name]))
        threads.append(thread)
        thread.start()
     
    live_dr.start_plotting()

    for thread in threads:
        thread.join()



