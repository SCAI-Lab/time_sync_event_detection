import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime as dt
import pytz

from collections import deque

from detectors.event_detector import *
from detectors.detector_zoo import get_model
from utils.plotting_utils import *
from utils.data_augmentation_utils import *

MYTZ = "Europe/Zurich"
tzinfo = pytz.timezone(MYTZ)

live_df = pd.DataFrame()
window = deque(maxlen=100)
anomaly_scores = []
ani = None


def plot_aug_mat(df, columns, model_name="IsolationForest"):
    """
    Function to plot live data streaming of sensomat data and detected anomaly/event.
    """
    global live_df, window, anomaly_scores, ani

    model = None

    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(tzinfo)
    
    interval = 1 / (df['time'].iloc[1] - df['time'].iloc[0])
    live_df = pd.DataFrame(columns=df.columns)
    
    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 10), sharex=True)
    columns[0] = 'Anomaly Score'
    color_map = get_cmap(len(columns) * 2)
    lines = []
    for i in range(len(columns)):
        line, = axes[i].plot([], [], label=columns[i], color=color_map(i), marker='.')
        lines.append(line)
        axes[i].set_ylabel(columns[i])
        axes[i].legend()
        axes[i].grid(True)
    axes[-1].set_xlabel('Time')

    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.3)
    plt.suptitle('Live Pressure Mat Data')

    def init(model_name=model_name):
        nonlocal model
        model = get_model(model_name)
        return model

    def update(frame):
        global live_df, window, anomaly_scores, ani
        nonlocal model

        if frame < len(df):
            new_data = df.iloc[frame:frame+1]
            live_df = pd.concat([live_df, new_data]).reset_index(drop=True)
            window.append(new_data[[*columns[1:]]].values[0])

        
        for i in range(1, len(lines)):
            lines[i].set_data(live_df['timestamp'][-100:], live_df[columns[i]][-100:])
        
        if len(window) == 100:
            predicted_score = detector_sk(window, model)
            anomaly_scores.append(predicted_score[-1])
        else:
            anomaly_scores.append(False) 

        lines[0].set_data(live_df['timestamp'][-100:], anomaly_scores[-100:])

        if not live_df.empty:
            for ax in (axes[1:]):
                ax.set_xlim(live_df['timestamp'][-100:].min(), live_df['timestamp'][-100:].max())
                ax.set_ylim(df[[*columns[1:]]].min().min(), df[[*columns[1:]]].max().max())

            axes[0].set_xlim(live_df['timestamp'][-100:].min(), live_df['timestamp'][-100:].max())
            axes[0].set_ylim(min(anomaly_scores[-100:]), max(anomaly_scores[-100:]))

        if frame >= len(df) - 1:
            ani.event_source.stop()

        return lines

    
    ani = FuncAnimation(fig, update, init_func=init, blit=False, interval=120)
    plt.tight_layout()
    plt.show()


def plot_mat(df, columns, model_name="IsolationForest"):
    """
    Function to plot live data streaming of sensomat data and detected anomaly/event.
    """
    global live_df, window, anomaly_scores, ani

    model = None

    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(tzinfo)
    
    interval = 1 / (df['time'].iloc[1] - df['time'].iloc[0])
    live_df = pd.DataFrame(columns=df.columns)
    
    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 10), sharex=True)
    columns[0] = 'Anomaly Score'
    color_map = get_cmap(len(columns))
    lines = []
    for i in range(len(columns)):
        line, = axes[i].plot([], [], label=columns[i], color=color_map(i), marker='.')
        lines.append(line)
        axes[i].set_ylabel(columns[i])
        axes[i].legend()
        axes[i].grid(True)
    axes[-1].set_xlabel('Time')

    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.3)
    plt.suptitle('Live Pressure Mat Data')

    def init(model_name=model_name):
        nonlocal model
        model = get_model(model_name)
        return model

    def update(frame):
        global live_df, window, anomaly_scores, ani
        nonlocal model

        if frame < len(df):
            new_data = df.iloc[frame:frame+1]
            live_df = pd.concat([live_df, new_data]).reset_index(drop=True)
            window.append(new_data[[*columns[1:]]].values[0])
        
        for i in range(1, len(lines)):
            lines[i].set_data(live_df['timestamp'][-100:], live_df[columns[i]][-100:])
        
        if len(window) == 100:
            predicted_score = detector_sk(window, model)
            anomaly_scores.append(predicted_score[-1])
        else:
            anomaly_scores.append(False) 

        lines[0].set_data(live_df['timestamp'][-100:], anomaly_scores[-100:])

        if not live_df.empty:
            for ax in (axes[1:]):
                ax.set_xlim(live_df['timestamp'][-100:].min(), live_df['timestamp'][-100:].max())
                ax.set_ylim(df[[*columns[1:]]].min().min(), df[[*columns[1:]]].max().max())

            axes[0].set_xlim(live_df['timestamp'][-100:].min(), live_df['timestamp'][-100:].max())
            axes[0].set_ylim(min(anomaly_scores[-100:]), max(anomaly_scores[-100:]))

        if frame >= len(df) - 1:
            ani.event_source.stop()

        return lines

    
    ani = FuncAnimation(fig, update, init_func=init, blit=False, interval=120)
    plt.tight_layout()
    # plt.show()

    # Save the animation
    ani.save('/Users/haozhu/Desktop/SCAI/live_data_streaming_detect.mp4', writer='ffmpeg', fps=10)
    plt.show()

if __name__ == "__main__":
    mat1_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/sensomative/user_223_2022-11-14_15-32-35-310[1].csv'
    mat2_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/sensomative/user_223b_2022-11-14_16-39-49-858[1].csv'
    cos_acc_csv  = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/cosinuss_ear_acc_x_acc_y_acc_z/K41C.9ZA0_2022-11-07_10-44-13_acc_x_acc_y_acc_z.csv'
    
    
    mat1 = pd.read_csv(mat1_csv)
    mat2 = pd.read_csv(mat2_csv)
    df = pd.concat([mat1, mat2], axis=0, ignore_index=True)
    # df = df.sort_values(by='time')

    # df = pd.read_csv(cos_acc_csv)

    df_idx = df[(df['time'] >= 1668438354 - 20) & (df['time'] <= 1668438354 + 2)].index
    df = df.loc[df_idx]

    # columns = ['time', 'acc_x', 'acc_y', 'acc_z']
    columns = ['time', 'device1_value4', 'device1_value5', 'device1_value9']


    df = df[[*columns]]
    print(df.shape)

    
    plot_mat(df, columns, model_name="LocalOutlierFactor")
