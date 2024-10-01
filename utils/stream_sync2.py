import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime as dt
import pytz
from collections import deque, defaultdict
import time
import threading
from typing import Dict, List, Any
import yaml

# from signal_sync import *
# from detectors.event_detector import *
# from detectors.detector_zoo import get_model
from evaluation.evaluation import *
# from utils.plotting_utils import *
from utils.data_augmentation_utils import *
# from utils.data_streaming import *
from src.DataReceiver import DataReceiver

# setup timezone for timestamp
MYTZ = "Europe/Zurich"
tzinfo = pytz.timezone(MYTZ)

# class DataReceiver:
#     def __init__(self, 
#                  sensors_config: Dict[str, Dict[str, Any]],
#                  plotting: bool,
#                  df_map: Dict[str, Any]):
#                 #  sensors: List[str], model_name: str, columns_map: Dict[str, List[str]], plotting: bool, window_size: int = 100):
#         self.sensors_config = sensors_config
#         self.sensors = list(sensors_config.keys())
#         self.df_map = df_map

#         self.window_size = {sensor: self.sensors_config[sensor]['window_size'] for sensor in self.sensors}
#         self.dataframe = {sensor: pd.DataFrame(columns=self.sensors_config[sensor]['columns']) for sensor in self.sensors}
#         self.window = {sensor: deque(maxlen=self.sensors_config[sensor]['window_size']) for sensor in self.sensors}
#         self.anomaly_scores = {sensor: [] for sensor in self.sensors}
#         self.ref_sensor = 'mat'
#         self.model = {sensor: get_model(self.sensors_config[sensor]) for sensor in self.sensors}
#         self.detector = {sensor: get_detector(self.sensors_config[sensor]['model_name']) for sensor in self.sensors}
#         self.lock = threading.Lock()  # Ensure thread-safe operations

#         # Plotting members
#         self.lines_map = defaultdict(list)
#         self.columns_map = {sensor: self.sensors_config[sensor]['columns'] for sensor in self.sensors}
#         self.prev_timeframe = 0
#         self.axes = None
#         self.fig = None
#         self.ani = None
#         self.all_data_streams_completed = threading.Event() ## ADD
#         self.animation_running = True
#         if plotting:
#             self.init_plot()

#     def start_plotting(self):
#         self.ani = FuncAnimation(self.fig, self.update_plot, blit=False, interval=200)
#         plt.tight_layout()
#         plt.show()

#     def init_plot(self):
#         fig, axes = plt.subplots(len(self.sensors) * 2, 1, figsize=(12, 10), sharex=False)
#         color_map = plt.get_cmap('tab10')

#         for i, (sensor_name, columns) in enumerate(self.columns_map.items()):
#             line, = axes[2 * i].plot([], [], label=f'Anomaly Score of {sensor_name}', color='red', marker='.')
#             self.lines_map[sensor_name].append((line))
#             axes[2 * i].set_ylabel('Anomaly Score')
#             axes[2 * i].legend()
#             axes[2 * i].grid(True)

#             for j, c in enumerate(columns[2:]):
#                 color_idx = (i * len(columns) + j) % color_map.N
#                 line_temp, = axes[2 * i + 1].plot([], [], label=c, color=color_map(color_idx), marker='.')
#                 self.lines_map[sensor_name].append((line_temp))

#             axes[2 * i + 1].set_xlabel('Time')
#             axes[2 * i + 1].set_ylabel(sensor_name)
#             axes[2 * i + 1].legend()
#             axes[2 * i + 1].grid(True)

#         plt.subplots_adjust(bottom=0.3)
#         plt.suptitle('Live Data Streaming with Time Drift')

#         self.axes = axes
#         self.fig = fig

#     def update_plot(self, frame):
#         if not self.animation_running:
#             return 
        
#         with self.lock:
#             for s in self.sensors:
#                 cur_a_score = self.anomaly_scores[s]
#                 # when anomaly score is changed, synchronize the time for all sensor
#                 if len(cur_a_score) >= 2 and cur_a_score[-1] != cur_a_score[-2]:
#                     # TODO: Ensure synchronization is correct
#                     self.dataframe, self.prev_timeframe = synchronization(self.ref_sensor, self.dataframe, self.prev_timeframe, frame)
#                     # print(f"shape: {self.dataframe['cos'].shape}  {len(self.anomaly_scores['cos'])}")
#             for i, (sensor_name, columns) in enumerate(self.columns_map.items()):
#                 if len(self.dataframe[sensor_name]) > 0:  # Ensure there is data
#                     self.lines_map[sensor_name][0].set_data(self.dataframe[sensor_name]['time'][-100:], self.anomaly_scores[sensor_name][-100:])
#                     for j, c in enumerate(columns[2:]):
#                         self.lines_map[sensor_name][j + 1].set_data(self.dataframe[sensor_name]['time'][-100:], self.dataframe[sensor_name][c][-100:])
                    
#                     if not self.dataframe[sensor_name].empty:
#                         self.axes[2 * i].set_xlim(self.dataframe[sensor_name]['time'][-100:].min(), self.dataframe[sensor_name]['time'][-100:].max())
#                         self.axes[2 * i].set_ylim(-1.2, 1.2)

#                         self.axes[2 * i + 1].set_xlim(self.dataframe[sensor_name]['time'][-100:].min(), self.dataframe[sensor_name]['time'][-100:].max())
#                         self.axes[2 * i + 1].set_ylim(self.dataframe[sensor_name][[*columns[2:]]][-100:].min().min() - 0.1, self.dataframe[sensor_name][[*columns[2:]]][-100:].max().max() + 0.1)
        
#         if self.all_data_streams_completed.is_set():
#             print("All data streams have completed. Stopping animation.")
#             self.animation_running = False
#             for s in self.sensors:
#                 cur_a_score = self.anomaly_scores[s]
#                 # TODO: Ensure synchronization is correct
#                 self.dataframe, self.prev_timeframe = synchronization(self.ref_sensor, self.dataframe, self.prev_timeframe, frame)
#             run_eval(self.dataframe, self.df_map)
            

#     def update_data(self, sensor_name: str, sensor_data: pd.DataFrame):
#         with self.lock:
#             sensor_data = sensor_data.T
#             sensor_data = sensor_data[[*self.columns_map[sensor_name]]]
#             self.dataframe[sensor_name] = pd.concat([self.dataframe[sensor_name], sensor_data], ignore_index=True)
#             self.window[sensor_name].append(sensor_data.iloc[0])

#     def event_detect(self, sensor_name: str):
#         with self.lock:
#             cur_window = self.window[sensor_name]
#             if len(cur_window) == self.window_size[sensor_name]:
#                 predicted_score = self.detector[sensor_name](cur_window, self.model[sensor_name])
#                 self.anomaly_scores[sensor_name].append(predicted_score[-1])
#             else:
#                 self.anomaly_scores[sensor_name].append(0)

def stream_data(sensor_data: pd.DataFrame, sampling_interval: float, init_delay: float, sensor_name: str, live_dr: DataReceiver, completion_event: threading.Event):
    time.sleep(init_delay)
    # for index in range(len(sensor_data)):
    #     row = sensor_data.loc[index:index]
    for _, row in sensor_data.iterrows():
        # print(f"*** beginning row: {row.dtypes}")
        row = row.to_frame()
        # row_df = pd.DataFrame([row], columns=sensor_data.columns)
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

    # sensors_config = config['sensors']
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

    df_map = {'mat': mat,
              'cos': cos_acc,
              'cor': cor_acc,
              'viva_acc': viva_acc}
    
    print(f"***** {viva_acc.shape}")
    live_dr = DataReceiver(sensors_config=sensors_config, plotting=plotting, df_map=df_map)
    completion_events = {sensors_config[sensor]['name']: threading.Event() for sensor in sensors_config}
    
    threads = []
    for sensor in sensors_config:
        name = sensors_config[sensor]['name']
        df = df_map[name]
        sampling_interval = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds() / len(df['time'])
        delay = (df['time'].min() - min(df['time'])).total_seconds()
        if name == 'cos':
            df = apply_time_drift(df, 10, 0.5)
        thread = threading.Thread(target=stream_data, args=(df, sampling_interval, delay, name, live_dr, completion_events[name]))
        threads.append(thread)
        thread.start()
     
    live_dr.start_plotting()

    # for event in completion_events.values():
    #     event.wait()
    print("All data streams have completed. Proceeding to evaluation.")
    run_eval(live_dr.dataframe, df_map)
    # plt.close('all') 

    for thread in threads:
        thread.join()
    # plot_thread.join()



# backup code
"""


    # Calculate the sampling intervals (in seconds)
    cos_acc_sampling_interval = (cos_acc['time'].iloc[-1] - cos_acc['time'].iloc[0]).total_seconds() / len(cos_acc['time'])
    mat_sampling_interval = (mat['time'].iloc[-1] - mat['time'].iloc[0]).total_seconds() / len(mat['time'])
    viva_hr_sampling_interval = (viva_hr['time'].iloc[-1] - viva_hr['time'].iloc[0]).total_seconds() / len(viva_hr['time'])
  

    start_times = [cos_acc['time'].min(), mat['time'].min(), viva_hr['time'].min()]
    earliest_start_time = min(start_times)
    print(f"start time: {start_times}")

    cos_acc_delay = (cos_acc['time'].min() - earliest_start_time).total_seconds()
    mat_delay = (mat['time'].min() - earliest_start_time).total_seconds()
    viva_hr_delay = (viva_hr['time'].min() - earliest_start_time).total_seconds()

    print(f"cos_acc_sampling_interval: {cos_acc_sampling_interval}")
    print(f"mat_sampling_interval: {mat_sampling_interval}")
    print(f"mat_sampling_interval: {viva_hr_sampling_interval}")
        
    print(f"cos_acc_delay: {cos_acc_delay}")
    print(f"mat_delay: {mat_delay}")
    print(f"mat_delay: {viva_hr_delay}")

    # data agumentation apply shift to cos acc data
    cos_acc_aug = apply_time_drift(cos_acc, 10, 0.5)

    mat_column = ['time', 'device1_value4', 'device1_value5', 'device1_value9']
    cos_column = ['time', 'acc_x', 'acc_y', 'acc_z']
    viva_hr_column = ['time', 'hr']

"""
