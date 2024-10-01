import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque, defaultdict
from typing import Dict, List, Any

from signal_sync import *
from detectors.event_detector import *
from detectors.detector_zoo import get_model
from evaluation.evaluation import *
from utils.plotting_utils import *
from utils.data_streaming import *


class DataReceiver:
    def __init__(self, 
                 sensors_config: Dict[str, Dict[str, Any]],
                 plotting: bool,
                 df_map: Dict[str, Any]):
        self.sensors_config = sensors_config
        self.sensors = list(sensors_config.keys())
        self.df_map = df_map

        self.window_size = {sensor: self.sensors_config[sensor]['window_size'] for sensor in self.sensors}
        self.dataframe = {sensor: pd.DataFrame(columns=self.sensors_config[sensor]['columns']) for sensor in self.sensors}
        self.window = {sensor: deque(maxlen=self.sensors_config[sensor]['window_size']) for sensor in self.sensors}
        self.anomaly_scores = {sensor: [] for sensor in self.sensors}
        self.ref_sensor = 'mat'
        self.model = {sensor: get_model(self.sensors_config[sensor]) for sensor in self.sensors}
        self.detector = {sensor: get_detector(self.sensors_config[sensor]['model_name']) for sensor in self.sensors}
        self.columns_map = {sensor: self.sensors_config[sensor]['columns'] for sensor in self.sensors}
        self.event_time = {sensor: [] for sensor in self.sensors}
        self.lock = threading.Lock()  # Ensure thread-safe operations

        # Plotting members
        self.lines_map = defaultdict(list)
        self.prev_timeframe = 0
        self.axes = None
        self.fig = None
        self.ani = None
        self.all_data_streams_completed = threading.Event() ## ADD
        self.animation_running = True
        if plotting:
            self.init_plot()

    def start_plotting(self):
        self.ani = FuncAnimation(self.fig, self.update_plot, blit=False, interval=200)
        plt.tight_layout()
        plt.show()

    def init_plot(self):
        fig, axes = plt.subplots(len(self.sensors) * 2, 1, figsize=(12, 10), sharex=False)
        color_map = plt.get_cmap('tab10')

        for i, (sensor_name, columns) in enumerate(self.columns_map.items()):
            line, = axes[2 * i].plot([], [], label=f'Anomaly Score of {sensor_name}', color='red', marker='.')
            self.lines_map[sensor_name].append((line))
            axes[2 * i].set_ylabel('Anomaly Score')
            axes[2 * i].legend()
            axes[2 * i].grid(True)

            for j, c in enumerate(columns[2:]):
                color_idx = (i * len(columns) + j) % color_map.N
                line_temp, = axes[2 * i + 1].plot([], [], label=c, color=color_map(color_idx), marker='.')
                self.lines_map[sensor_name].append((line_temp))

            axes[2 * i + 1].set_xlabel('Time')
            axes[2 * i + 1].set_ylabel(sensor_name)
            axes[2 * i + 1].legend()
            axes[2 * i + 1].grid(True)

        plt.subplots_adjust(bottom=0.3)
        plt.suptitle('Live Data Streaming with Time Drift')

        self.axes = axes
        self.fig = fig

    def update_plot(self, frame):
        if not self.animation_running:
            return 
        
        with self.lock:
            for s in self.sensors:
                cur_a_score = self.anomaly_scores[s]
                # when anomaly score is changed, synchronize the time for all sensor
                if len(cur_a_score) >= 2 and cur_a_score[-1] != cur_a_score[-2]:
                    # TODO: Ensure synchronization is correct
                    self.dataframe, self.prev_timeframe = synchronization(self.ref_sensor, self.dataframe, self.prev_timeframe, frame)

            for i, (sensor_name, columns) in enumerate(self.columns_map.items()):
                if len(self.dataframe[sensor_name]) > 0:  # Ensure there is data
                    self.lines_map[sensor_name][0].set_data(self.dataframe[sensor_name]['time'][-100:], self.anomaly_scores[sensor_name][-100:])
                    for j, c in enumerate(columns[2:]):
                        self.lines_map[sensor_name][j + 1].set_data(self.dataframe[sensor_name]['time'][-100:], self.dataframe[sensor_name][c][-100:])
                    
                    if not self.dataframe[sensor_name].empty:
                        self.axes[2 * i].set_xlim(self.dataframe[sensor_name]['time'][-100:].min(), self.dataframe[sensor_name]['time'][-100:].max())
                        self.axes[2 * i].set_ylim(-1.2, 1.2)

                        self.axes[2 * i + 1].set_xlim(self.dataframe[sensor_name]['time'][-100:].min(), self.dataframe[sensor_name]['time'][-100:].max())
                        self.axes[2 * i + 1].set_ylim(self.dataframe[sensor_name][[*columns[2:]]][-100:].min().min() - 0.1, self.dataframe[sensor_name][[*columns[2:]]][-100:].max().max() + 0.1)
        
        if self.all_data_streams_completed.is_set():
            print("All data streams have completed. Stopping animation.")
            self.animation_running = False
            for s in self.sensors:
                cur_a_score = self.anomaly_scores[s]
                # TODO: Ensure synchronization is correct
                self.dataframe, self.prev_timeframe = synchronization(self.ref_sensor, self.dataframe, self.prev_timeframe, frame)
            run_eval(self)
            

    def update_data(self, sensor_name: str, sensor_data: pd.DataFrame):
        with self.lock:
            sensor_data = sensor_data.T
            sensor_data = sensor_data[[*self.columns_map[sensor_name]]]
            self.dataframe[sensor_name] = pd.concat([self.dataframe[sensor_name], sensor_data], ignore_index=True)
            self.window[sensor_name].append(sensor_data.iloc[0])

    def event_detect(self, sensor_name: str):
        with self.lock:
            cur_window = self.window[sensor_name]
            # print(f"{sensor_name}: {cur_window[-1]['timestamp']}")
            if len(cur_window) == self.window_size[sensor_name]:
                predicted_score = self.detector[sensor_name](cur_window, self.model[sensor_name])
                self.anomaly_scores[sensor_name].append(predicted_score[-1])
                if predicted_score[-1] == 1:
                    self.event_time[sensor_name].append(cur_window[-1]['time'])
                    # if len(self.anomaly_scores[sensor_name]) >= 2 and self.anomaly_scores[sensor_name][-1] != self.anomaly_scores[sensor_name][-2]:
                    #     # TODO: Ensure synchronization is correct
                    #     self.dataframe, self.prev_timeframe = synchronization(self.ref_sensor, self.dataframe, self.prev_timeframe, frame)

            else:
                self.anomaly_scores[sensor_name].append(0)
