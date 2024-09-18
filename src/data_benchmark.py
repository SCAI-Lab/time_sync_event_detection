"""
This file clean the raw data AND convert them starting from 1970.01.01
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import pytz

from utils.files_utils import *
from utils.time_utils import *


MYTZ = "Europe/Zurich"
tzinfo = pytz.timezone(MYTZ)


def data_cleaning(target_rec, label_guide_xlsx, saved_dir):
    """
    @param target_rec: some recordings and the corresponding dates to benchmarking
    @param label_guide_xlsx: the xlsx sheet that records the video start/end time and correspond frames,
    useful to calculate frame rate and the exactly sync time 
    @param saved_dir: the output path to saved the cleaned dataframes into csv files
    """

    for user in target_rec.keys():
        user_id = user.split('/')[-1]
        YEAR, MONTH, DAY = target_rec[user]
        
        user_event = label_guide_xlsx[label_guide_xlsx['User'] == user_id].iloc[0]

        start_frame = user_event['first_frame_of_second_START']
        start_time  = user_event['frame_laptop_time_START']
        end_frame   = user_event['first_frame_of_second_END']
        end_time    = user_event['frame_laptop_time_END']

        sync_frame_start = user_event['VideoFrame_Touchpad_START_1']
        sync_frame_end = user_event['VideoFrame_Touchpad_END_3']
        
        rec_date  = datetime(YEAR, MONTH, DAY) 
        start_time_comb = datetime.combine(rec_date, start_time)
        end_time_comb   = datetime.combine(rec_date, end_time)
        
        duration_seconds = (end_time_comb - start_time_comb).total_seconds()
        total_frames = end_frame - start_frame

        FS = total_frames / duration_seconds
        
        accurate_sec_sync_start = frame2sec(sync_frame_start, start_frame, FS)
        accurate_datetime_sync_start = sec2datetime(accurate_sec_sync_start - 5, start_time_comb)
        unix_start = int(accurate_datetime_sync_start.timestamp())

        accurate_sec_sync_end = frame2sec(sync_frame_end, start_frame, FS)
        accurate_datetime_sync_end = sec2datetime(accurate_sec_sync_end + 15, start_time_comb)
        unix_end = int(accurate_datetime_sync_end.timestamp())
        
        print('user: {}, start_time: {}, end_time: {}, FS: {}'.format(user_id, 
                                                                    accurate_datetime_sync_start, 
                                                                    accurate_datetime_sync_end, 
                                                                    FS))
        
        # Set Sensomat Signal to be ground truth
        gt_rec_path = os.path.join(user, 'sensomative')
        gt_mat_rec = None
        for file in os.listdir(gt_rec_path):
            filepath = os.path.join(gt_rec_path, file)
            if filepath.endswith('.csv'):
                if gt_mat_rec is None:
                    gt_mat_rec = pd.read_csv(filepath)
                else: 
                    temp = pd.read_csv(filepath)
                    gt_mat_rec = pd.concat([gt_mat_rec, temp], axis=0, ignore_index=True)

        
        if gt_mat_rec is None:
            raise ValueError("No Sensomat Data Read.")
        
        gt_mat_rec = gt_mat_rec.sort_values(by='time')
        gt_mat_rec_idx = gt_mat_rec[(gt_mat_rec['time'] >= unix_start) & (gt_mat_rec['time'] <= unix_end)].index
        gt_mat_rec = gt_mat_rec.loc[gt_mat_rec_idx]

        gt_senso_time = gt_mat_rec['time'].to_numpy()
        gt_starting_time = gt_senso_time[0]
        
        print(gt_senso_time[-1], gt_senso_time[0])
        print(unix_end, unix_start)
        
        for subdir in os.listdir(user):
            subdir_path = os.path.join(user, subdir)
            if os.path.isdir(subdir_path):
                csv_df = read_csv(subdir_path)
                print('\t ', user_id + '/' + subdir)

                if 'time' not in csv_df.columns: 
                    print('\t\t time not in column')
                    continue

                csv_df = csv_df.sort_values(by='time')
                csv_df_idx = csv_df[(csv_df['time'] >= unix_start) & (csv_df['time'] <= unix_end)].index
                csv_df = csv_df.loc[csv_df_idx]

                if csv_df.size == 0:
                    print('\t not corresponding time found')
                    continue

                csv_df['time'] = csv_df['time'] - gt_starting_time
                
                saved_folders = os.path.join(saved_dir, user_id, subdir)
                if not os.path.exists(saved_folders):
                    os.makedirs(saved_folders)
                
                saved_csv_path = os.path.join(saved_folders, str(DAY) + '_' + str(MONTH) + '_' + str(YEAR) + '_' + subdir + '.csv')
                csv_df.to_csv(saved_csv_path, index=False)
                

                csv_df_time = csv_df['time'].to_numpy()

                print('\t\t start time: ', datetime.fromtimestamp(csv_df_time[0]), ' unix time: ', csv_df_time[0])
                print('\t\t end time  : ', datetime.fromtimestamp(csv_df_time[-1]), ' unix time: ', csv_df_time[-1])
                print('\t\t FS: ', (csv_df_time[-1] - csv_df_time[0]) / len(csv_df_time))
    
    
    
    

if __name__ == '__main__':
    sensor_str = set(('cosinuss', 'sensomative', 'vivalnk', 'corsano', 'zurichmove'))

    target_rec = {'/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-103': (2022, 11, 8),
                '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-188': (2022, 11, 16),
                '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223': (2022, 11, 14),
                '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-489': (2022, 11, 7)}

    label_guide_xlsx = pd.read_excel('/Users/haozhu/Desktop/SCAI/sensei_v2_parsed/Sensei-V2 - Modified-Labels.xlsx', sheet_name='Tabelle1')

    saved_dir = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2_filtered'

    data_cleaning(target_rec=target_rec, label_guide_xlsx=label_guide_xlsx, saved_dir=saved_dir)
    print('**** Finish ****')

