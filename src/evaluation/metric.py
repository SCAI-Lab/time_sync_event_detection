import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pandas as pd
import numpy as np


def RMSE(corrected_df, gt_df):
    corrected_time = corrected_df['time'].reset_index(drop=True)
    gt_time = gt_df['time'].reset_index(drop=True)
    
    time_error = (corrected_time - gt_time).dt.total_seconds()
    # print(f"time_error: {time_error}")
    rmse = np.sqrt((time_error ** 2).mean())
    return rmse


def MAE(corrected_df, gt_df):
    corrected_time = corrected_df['time'].reset_index(drop=True)
    gt_time = gt_df['time'].reset_index(drop=True)
    
    time_error = (corrected_time - gt_time).dt.total_seconds()
    mae = time_error.abs().mean()
    return mae

