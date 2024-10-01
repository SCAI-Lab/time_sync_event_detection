import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pytz
MYTZ = "Europe/Zurich"
tzinfo = pytz.timezone(MYTZ)

from metric import *
from DataReceiver import DataReceiver


def run_eval(dr: DataReceiver):
    updated_dfs = dr.dataframe
    gt_dfs = dr.df_map
    for sensor_name in updated_dfs.keys():
        print(f"sensor_name: {sensor_name}")

        updated_dfs[sensor_name]['time'] = pd.to_datetime(updated_dfs[sensor_name]['time'], errors='coerce')
        updated_dfs[sensor_name]['time'] = updated_dfs[sensor_name]['time'].dt.tz_convert(tzinfo)
        rmse = RMSE(updated_dfs[sensor_name], gt_dfs[sensor_name])
        mae = MAE(updated_dfs[sensor_name], gt_dfs[sensor_name])

        print(f"\t RMSE of Sensor {sensor_name}: {rmse}")
        print(f"\t MAE of Sensor {sensor_name}: {mae}")

        print(f"\t # of Detected Events: {sum(dr.anomaly_scores[sensor_name])}")

        # print(f"\t detected event time: {dr.event_time[sensor_name]}")
    