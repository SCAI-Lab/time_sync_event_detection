import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import datetime as dt
import pytz

from collections import deque

from signal_sync import *
from detectors.event_detector import *
from detectors.detector_zoo import get_model
from utils.plotting_utils import *
from utils.data_augmentation_utils import *


MYTZ = "Europe/Zurich"
tzinfo = pytz.timezone(MYTZ)

live_df = pd.DataFrame()
live_df_drift = pd.DataFrame()
window = deque(maxlen=100)
window_drift = deque(maxlen=100)
anomaly_scores = []
anomaly_scores_drift = []
ani = None


def plot_mat(df, columns, model_name="IsolationForest"):
    """
    Function to plot live data streaming of sensomat data and detected anomaly/event.
    """
    global live_df, live_df_drift, window, window_drift, anomaly_scores, ani

    model = None
    prev_timeframe = 0

    # df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    # df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(tzinfo)
    
    df_augmented = apply_time_drift(df, 10, 0.5)

    # interval = 1 / (df['time'].iloc[1] - df['time'].iloc[0])
    live_df = pd.DataFrame(columns=df.columns)
    live_df_drift = pd.DataFrame(columns=df_augmented.columns)
    
    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 10), sharex=True)
    color_map = plt.get_cmap('tab10')
    lines = []
    columns[0] = 'Anomaly Score'

    line, = axes[0].plot([], [], label=columns[0], color=color_map(0), marker='.')
    lines.append((line))
    axes[0].set_ylabel(columns[0])
    axes[0].legend()
    axes[0].grid(True)
    
    for i in range(1, len(columns)):
        line1, = axes[i].plot([], [], label=f"Original {columns[i]}", color=color_map(2*i), marker='.')
        line2, = axes[i].plot([], [], label=f"Drifted {columns[i]}", color=color_map(2*i + 1), linestyle='--', marker='.')
        
        lines.append((line1, line2))
        
        axes[i].set_ylabel(columns[i])
        axes[i].legend()
        axes[i].grid(True)
    
    axes[-1].set_xlabel('Time')

    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.3)
    plt.suptitle('Live Data Streaming with Time Drift')

    def init(model_name=model_name):
        nonlocal model
        model = get_model(model_name)
        return model

    def update(frame):
        global live_df, live_df_drift, window, window_drift, anomaly_scores, ani
        nonlocal model, prev_timeframe

        if frame < len(df):
            new_data = df.iloc[frame:frame+1]
            new_data_drift = df_augmented.iloc[frame:frame+1]
            # print(f"new_data: {new_data}, \n new_data_drift: {new_data_drift}")
            
            live_df = pd.concat([live_df, new_data]).reset_index(drop=True)
            live_df_drift = pd.concat([live_df_drift, new_data_drift]).reset_index(drop=True)
            
            # window.append(new_data[[*columns[1:]]].values[0])
            # window_drift.append(new_data_drift[[*columns[1:]]].values[0])
            window.append(new_data[[*columns[1:]]].iloc[0])
            print(new_data[[*columns[1:]]].iloc[0])
            window_drift.append(new_data_drift[[*columns[1:]]].iloc[0])

        if len(window) == 100:
            predicted_score = detector_sk(window, model)
            anomaly_scores.append(predicted_score[-1])
        else:
            anomaly_scores.append(False)

        if anomaly_scores[-1] != anomaly_scores[-2]:
            print(f"prev_timeframe: {prev_timeframe}, cur frame: {frame}")
            # print(live_df['timestamp'].iloc[0])
            [live_df_drift], prev_timeframe = synchronization(live_df, [live_df_drift], prev_timeframe, frame)
        


        # plotting
        for i in range(1, len(lines)):
            lines[i][0].set_data(live_df['time'][-100:], live_df[columns[i]][-100:])
            lines[i][1].set_data(live_df_drift['time'][-100:], live_df_drift[columns[i]][-100:])
        
        lines[0].set_data(live_df['time'][-100:], anomaly_scores[-100:])

        if not live_df.empty:
            for ax in axes:
                ax.set_xlim(live_df['time'][-100:].min(), live_df['time'][-100:].max())
                ax.set_ylim(df[[*columns[1:]]].min().min(), df[[*columns[1:]]].max().max())

            axes[0].set_xlim(live_df['time'][-100:].min(), live_df['time'][-100:].max())
            axes[0].set_ylim(min(anomaly_scores[-100:]), max(anomaly_scores[-100:]))

        if frame >= len(df) - 1:
            ani.event_source.stop()

        return

    ani = FuncAnimation(fig, update, init_func=init, blit=False, interval=150)
    plt.tight_layout()
    plt.show()
    
    # # Save the animation
    # ani.save('/Users/haozhu/Desktop/SCAI/live_data_streaming_drift.mp4', writer='ffmpeg', fps=10)
    

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

    # columns = ['time', 'acc_x', 'acc_y', 'acc_z']
    columns = ['time', 'device1_value4', 'device1_value5', 'device1_value9']

    df_mat = mat[[*columns]]
    print(df_mat.shape)

    plot_mat(df_mat, columns, model_name="LocalOutlierFactor")



# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import datetime as dt
# import pytz
# import cv2

# from collections import deque

# from signal_sync import *
# from detectors.event_detector import *
# from detectors.detector_zoo import get_model
# from utils.plotting_utils import *
# from utils.data_augmentation_utils import *


# MYTZ = "Europe/Zurich"
# tzinfo = pytz.timezone(MYTZ)

# live_df = pd.DataFrame()
# live_df_drift = pd.DataFrame()
# window = deque(maxlen=100)
# window_drift = deque(maxlen=100)
# anomaly_scores = []
# anomaly_scores_drift = []
# ani = None


# def plot_mat(df, columns, model_name="IsolationForest"):
#     """
#     Function to plot live data streaming of sensomat data and detected anomaly/event.
#     """
#     global live_df, live_df_drift, window, window_drift, anomaly_scores, ani

#     model = None
#     prev_timeframe = 0

#     df['timestamp'] = pd.to_datetime(df['time'], unit='s')
#     df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(tzinfo)
    
#     df_augmented = apply_time_drift(df, 10, 0.5)

#     interval = 1 / (df['time'].iloc[1] - df['time'].iloc[0])
#     live_df = pd.DataFrame(columns=df.columns)
#     live_df_drift = pd.DataFrame(columns=df_augmented.columns)
    
#     fig, axes = plt.subplots(len(columns), 1, figsize=(12, 10), sharex=True)
#     color_map = plt.get_cmap('tab10')
#     lines = []
#     columns[0] = 'Anomaly Score'

#     line, = axes[0].plot([], [], label=columns[0], color=color_map(0), marker='.')
#     lines.append((line))
#     axes[0].set_ylabel(columns[0])
#     axes[0].legend()
#     axes[0].grid(True)
    
#     for i in range(1, len(columns)):
#         line1, = axes[i].plot([], [], label=f"Original {columns[i]}", color=color_map(2*i), marker='.')
#         line2, = axes[i].plot([], [], label=f"Drifted {columns[i]}", color=color_map(2*i + 1), linestyle='--', marker='.')
        
#         lines.append((line1, line2))
        
#         axes[i].set_ylabel(columns[i])
#         axes[i].legend()
#         axes[i].grid(True)
    
#     axes[-1].set_xlabel('Time')

#     plt.xticks(rotation=45, ha='right')
#     plt.subplots_adjust(bottom=0.3)
#     plt.suptitle('Live Data Streaming with Time Drift')

#     def init(model_name=model_name):
#         nonlocal model
#         model = get_model(model_name)
#         return model

#     def update(frame):
#         global live_df, live_df_drift, window, window_drift, anomaly_scores, ani
#         nonlocal model, prev_timeframe

#         if frame < len(df_augmented):
#             new_data = df.iloc[frame:frame+1]
#             new_data_drift = df_augmented.iloc[frame:frame+1]
            
#             live_df = pd.concat([live_df, new_data]).reset_index(drop=True)
#             live_df_drift = pd.concat([live_df_drift, new_data_drift]).reset_index(drop=True)
            
#             window.append(new_data[[*columns[1:]]].values[0])
#             window_drift.append(new_data_drift[[*columns[1:]]].values[0])
        
#         if len(window) == 100:
#             predicted_score = detector_sk(window, model)
#             anomaly_scores.append(predicted_score[-1])
#         else:
#             anomaly_scores.append(False)

#         if anomaly_scores[-1]:
#             [live_df_drift], prev_timeframe = synchronization(live_df, [live_df_drift], prev_timeframe, frame)
        
#         print(f"frame: {frame}")

#          # plotting
#         for i in range(1, len(lines)):
#             lines[i][0].set_data(live_df['timestamp'][-100:], live_df[columns[i]][-100:])
#             lines[i][1].set_data(live_df_drift['timestamp'][-100:], live_df_drift[columns[i]][-100:])
        
#         lines[0].set_data(live_df['timestamp'][-100:], anomaly_scores[-100:])

#         if not live_df.empty:
#             for ax in axes:
#                 ax.set_xlim(live_df['timestamp'][-100:].min(), live_df['timestamp'][-100:].max())
#                 ax.set_ylim(df[[*columns[1:]]].min().min(), df[[*columns[1:]]].max().max())

#             axes[0].set_xlim(live_df['timestamp'][-100:].min(), live_df['timestamp'][-100:].max())
#             axes[0].set_ylim(min(anomaly_scores[-100:]), max(anomaly_scores[-100:]))

#         if frame >= len(df_augmented) - 1:
#             ani.event_source.stop()

#         # return [line_pair for line_pair in lines]
#         return

#     ani = FuncAnimation(fig, update, init_func=init, blit=False, interval=150)
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     mat1_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/sensomative/user_223_2022-11-14_15-32-35-310[1].csv'
#     mat2_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/sensomative/user_223b_2022-11-14_16-39-49-858[1].csv'
#     cos_acc_csv  = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/cosinuss_ear_acc_x_acc_y_acc_z/K41C.9ZA0_2022-11-07_10-44-13_acc_x_acc_y_acc_z.csv'
    
    
#     mat1 = pd.read_csv(mat1_csv)
#     mat2 = pd.read_csv(mat2_csv)
#     df = pd.concat([mat1, mat2], axis=0, ignore_index=True)
#     # df = df.sort_values(by='time')

#     # df = pd.read_csv(cos_acc_csv)

#     df_idx = df[(df['time'] >= 1668438354 - 20) & (df['time'] <= 1668438354 + 2)].index
#     df = df.loc[df_idx]

#     # columns = ['time', 'acc_x', 'acc_y', 'acc_z']
#     columns = ['time', 'device1_value4', 'device1_value5', 'device1_value9']

#     df = df[[*columns]]
#     print(df.shape)

#     plot_mat(df, columns, model_name="LocalOutlierFactor")
