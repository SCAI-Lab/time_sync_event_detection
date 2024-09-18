import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import pytz
MYTZ = "Europe/Zurich"
tzinfo = pytz.timezone(MYTZ)

from metric import *

def run_eval(updated_dfs, gt_dfs):
    for sensor_name in gt_dfs.keys():
        print(f"sensor_name: {sensor_name}")

        updated_dfs[sensor_name]['time'] = pd.to_datetime(updated_dfs[sensor_name]['time'], errors='coerce')
        updated_dfs[sensor_name]['time'] = updated_dfs[sensor_name]['time'].dt.tz_convert(tzinfo)
        rmse = RMSE(updated_dfs[sensor_name], gt_dfs[sensor_name])
        # mae = MAE(updated_dfs[sensor_name], gt_dfs[sensor_name])

        print(f"RMSE of Sensor {sensor_name}: {rmse}")
        # print(f"MAE of Sensor {sensor_name}: {mae}")
    