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
from typing import List, Dict

from signal_sync import *
from detectors.event_detector import *
from detectors.detector_zoo import get_model
from utils.plotting_utils import *
from utils.data_augmentation_utils import *
from utils.data_streaming import *


MYTZ = "Europe/Zurich"
tzinfo = pytz.timezone(MYTZ)

mat_df = pd.DataFrame()
live_df_drift = pd.DataFrame()
window = deque(maxlen=100)
anomaly_scores = []
ani = None

class DataReceiver:
    def __init__(self,
                 sensors : List[str],
                 model_name : str,
                 columns_map : Dict[str, List[str]],
                 plotting : bool,
                 window_size : int = 100) -> None:
    
        # data storage member
        self.sensors = sensors # number of sensor
        self.window_size = window_size
        self.dataframe = {sensor : pd.DataFrame(columns=columns_map[sensor]) for sensor in sensors}
        self.window = {sensor : deque(maxlen=window_size) for sensor in sensors}
        self.anomaly_scores = {sensor : [] for sensor in sensors}
        self.ref_sensor = 'mat'

        self.model = get_model(model_name)

        # plot member
        self.lines_map = defaultdict(list)
        self.columns_map = columns_map
        self.prev_timeframe = 0
        self.axes = None
        self.fig  = None
        self.ani  = None
        if plotting:
            self.init_plot()

    def thread_update(self):
        self.ani = FuncAnimation(self.fig, self.update_plot, blit=False, interval=150)
        plt.tight_layout()
        plt.show()

    def init_plot(self):
        """
        self.line_map is dict of list
            key: sensor name; 
            value: 0 idx: anomaly score; 1 idx and the following: columns (for example: acc_x, acc_y, acc_z)
        """
        fig, axes = plt.subplots(len(self.sensors) * 2, 1, figsize=(12, 10), sharex=True)
        color_map = plt.get_cmap('tab10')


        for i, (sensor_name, columns) in enumerate(self.columns_map.items()):
            # even idx always store anomaly score of each sensor
            line, = axes[2 * i].plot([], [], label=f'Anomaly Score of {sensor_name}', color='red', marker='.')
            self.lines_map[sensor_name].append((line))
            axes[2 * i].set_ylabel('Anomaly Score')
            axes[2 * i].legend()
            axes[2 * i].grid(True)

            # odd idx always store data of each sensor
            # ignore columns[1] which is time
            for j, c in enumerate(columns[1:]):
                color_idx = (i * len(columns) + j) % color_map.N
                line_temp, = axes[2 * i + 1].plot([], [], label=c, color=color_map(color_idx), marker='.')
                self.lines_map[sensor_name].append((line_temp))
            
            axes[2 * i + 1].set_ylabel(columns[0][:-1])
            axes[2 * i + 1].legend()
            axes[2 * i + 1].grid(True)
        
        axes[-1].set_xlabel('Time')

        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.3)
        plt.suptitle('Live Data Streaming with Time Drift')

        self.axes = axes
        self.fig  = fig
    
    def update_plot(self, frame):
        for s in self.sensors:
            cur_a_score = self.anomaly_scores[s]
            if cur_a_score[-1] != cur_a_score[-2]:
                print(f"prev_timeframe: {prev_timeframe}, cur frame: {frame}")
                # print(live_df['timestamp'].iloc[0])
                [self.dataframe['cos']], prev_timeframe = synchronization(self.dataframe[self.ref_sensor], [self.dataframe['cos']], prev_timeframe, frame)
        
        # plotting
        for i, (sensor_name, columns) in enumerate(self.columns_map.items()):
            self.lines_map[sensor_name][0].set_data(self.dataframe[sensor_name]['time'][-100:], self.anomaly_scores[sensor_name][-100:])

            for j, c in enumerate(columns[1:]):
                self.lines_map[sensor_name][j + 1].set_data(self.dataframe[sensor_name]['time'][-100:], self.dataframe[sensor_name][c][-100:])
            
            if not self.dataframe[sensor_name].empty:
                self.axes[2 * i].set_xlim(self.dataframe[sensor_name]['time'][-100:].min(), self.dataframe[sensor_name]['time'][-100:].max())
                self.axes[2 * i].set_ylim(self.anomaly_scores[sensor_name][-100:].min(), self.anomaly_scores[sensor_name][-100:].max())

                self.axes[2 * i + 1].set_xlim(self.dataframe[sensor_name]['time'][-100:].min(), self.dataframe[sensor_name]['time'][-100:].max())
                self.axes[2 * i + 1].set_ylim(self.dataframe[sensor_name][[*columns[1:]]][-100:].min().min(), self.dataframe[sensor_name][[*columns[1:]]][-100:].max().max())



        # for i in range(1, len(lines)):
        #     lines[i][0].set_data(live_df['timestamp'][-100:], live_df[columns[i]][-100:])
        #     lines[i][1].set_data(live_df_drift['timestamp'][-100:], live_df_drift[columns[i]][-100:])
        
        # lines[0].set_data(live_df['timestamp'][-100:], anomaly_scores[-100:])

        # if not live_df.empty:
        #     for ax in axes:
        #         ax.set_xlim(live_df['timestamp'][-100:].min(), live_df['timestamp'][-100:].max())
        #         ax.set_ylim(df[[*columns[1:]]].min().min(), df[[*columns[1:]]].max().max())

        #     axes[0].set_xlim(live_df['timestamp'][-100:].min(), live_df['timestamp'][-100:].max())
        #     axes[0].set_ylim(min(anomaly_scores[-100:]), max(anomaly_scores[-100:]))

        # if frame >= len(df) - 1:
        #     ani.event_source.stop()

        return
    
    # TODO: a more robust way to update (which also depends on how data is transmitted)
    def update_data(self,
                    sensor_name : str,
                    sensor_data : pd.DataFrame):
        sensor_data = sensor_data.T
        sensor_data = sensor_data[[*self.columns_map[sensor_name]]]
        self.dataframe[sensor_name] = pd.concat([self.dataframe[sensor_name], sensor_data], ignore_index=True)
        self.window[sensor_name].append(sensor_data[[*(self.columns_map[sensor_name][1:])]].values[0])

        # print(f"\t\t window new data: {sensor_data[[*(self.columns_map[sensor_name][1:])]].values[0]}")
        print(f"data frame: \n{self.dataframe[sensor_name]}")

    def event_detect(self, sensor_name : str):
        cur_window = self.window[sensor_name]
        if len(cur_window) == self.window_size:
            predicted_score = detector_sk(cur_window, self.model)
            self.anomaly_scores[sensor_name].append(predicted_score[-1])
        else:
            self.anomaly_scores[sensor_name].append(False)
    



def stream_data(
        sensor_data : pd.DataFrame, 
        sampling_interval : float, 
        init_delay: float, 
        sensor_name : str, 
        live_dr : DataReceiver):
    
    time.sleep(init_delay)
    for _, row in sensor_data.iterrows():
        row = row.to_frame()
        live_dr.update_data(sensor_name, row)
        live_dr.event_detect(sensor_name)
        live_dr.thread_update()
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

    # data agumentation apply shift to cos acc data
    cos_acc_aug = apply_time_drift(cos_acc, 10, 0.5)

    mat_column = ['time', 'device1_value4', 'device1_value5', 'device1_value9']
    cos_column = ['time', 'acc_x', 'acc_y', 'acc_z']

    columns = {'mat': mat_column,
               'cos': cos_column}
    
    live_dr = DataReceiver(sensors=['mat', 'cos'], model_name='LocalOutlierFactor', columns_map=columns,
                           plotting=True)
    
    # Create threads for each sensor
    cos_acc_thread = threading.Thread(target=stream_data, args=(cos_acc_aug, cos_acc_sampling_interval, cos_acc_delay, 'cos', live_dr))
    mat_thread     = threading.Thread(target=stream_data, args=(mat, mat_sampling_interval, mat_delay, 'mat', live_dr))

    # Start the threads
    cos_acc_thread.start()
    mat_thread.start()

    plt.show()


# def stream_data(
#         sensor_data : pd.DataFrame, 
#         sampling_interval : float, 
#         init_delay: float, 
#         sensor_name : str, 
#         columns : dict, 
#         fig : plt.Figure, 
#         axes : plt.Axes, 
#         lines : dict, 
#         model : str, 
#         prev_timeframe : int):
    
#     time.sleep(init_delay)
#     for _, row in sensor_data.iterrows():
#         # Convert the row to DataFrame to match the expected input in plot_update
#         # new_data = pd.DataFrame([row])
#         # print(type(row))
#         row = row.to_frame()
#         print(row.index)
#         print(row.T)
#         # print(f"{sensor_name} \n *** data: {row[:3]} \n *** at {row['time']}")
#         # prev_timeframe = plot_update(row, sensor_name, columns, fig, axes, lines, model, prev_timeframe)
#         time.sleep(sampling_interval)