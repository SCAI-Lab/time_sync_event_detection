from nixtla import NixtlaClient
import pandas as pd
nixtla_client = NixtlaClient()
from nixtla.utils import in_colab

import pytz

MYTZ = "Europe/Zurich"
tzinfo = pytz.timezone(MYTZ)


IN_COLAB = in_colab()
print(f"\n\nXXXXX {IN_COLAB}")

nixtla_client.validate_api_key()

mat1_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/sensomative/user_223_2022-11-14_15-32-35-310[1].csv'
mat2_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/sensomative/user_223b_2022-11-14_16-39-49-858[1].csv'
cos_acc_csv  = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/cosinuss_ear_acc_x_acc_y_acc_z/K41C.9ZA0_2022-11-07_10-44-13_acc_x_acc_y_acc_z.csv'
viva_hr1_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/vivalnk_vv330_heart_rate/20221114/20221114_1400.csv'
viva_hr2_csv = '/Users/haozhu/Desktop/SCAI/SCAI-SENSEI-V2/sensei-223/vivalnk_vv330_heart_rate/20221114/20221114_1500.csv'


# Read CSV files
mat1 = pd.read_csv(mat1_csv)
mat2 = pd.read_csv(mat2_csv)
mat = pd.concat([mat1, mat2], axis=0, ignore_index=True)
mat = mat.sort_values(by='time')
viva_hr1 = pd.read_csv(viva_hr1_csv)
viva_hr2 = pd.read_csv(viva_hr2_csv)
viva_hr = pd.concat([viva_hr1, viva_hr2], axis=0, ignore_index=True)
cos_acc = pd.read_csv(cos_acc_csv)

# selected time
df_idx = mat[(mat['time'] >= 1668438354 - 20) & (mat['time'] <= 1668438354 + 2)].index
mat = mat.loc[df_idx]

df_idx = cos_acc[(cos_acc['time'] >= 1668438354 - 20) & (cos_acc['time'] <= 1668438354 + 2)].index
cos_acc = cos_acc.loc[df_idx]

df_idx = viva_hr[(viva_hr['time'] >= 1668438354 - 20) & (viva_hr['time'] <= 1668438354 + 2)].index
viva_hr = viva_hr.loc[df_idx]

cos_acc['timestamp'] = cos_acc['time']
mat['timestamp'] = mat['time']
viva_hr['timestamp'] = viva_hr['time']

cos_acc['time'] = pd.to_datetime(cos_acc['time'], unit='s')
mat['time'] = pd.to_datetime(mat['time'], unit='s')
viva_hr['time'] = pd.to_datetime(viva_hr['time'], unit='s')

mat['time'] = mat['time'].dt.tz_localize('UTC').dt.tz_convert(tzinfo)
cos_acc['time'] = cos_acc['time'].dt.tz_localize('UTC').dt.tz_convert(tzinfo)
viva_hr['time'] = viva_hr['time'].dt.tz_localize('UTC').dt.tz_convert(tzinfo)

print(mat[['time', 'device1_value4', 'device1_value5', 'device1_value9']])

# df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity-short-with-ex-vars.csv')

# # Detect anomalies
anomalies_df = nixtla_client.detect_anomalies(
    df=mat[['time', 'device1_value4', 'device1_value5', 'device1_value9']],
    freq='ms',
    time_col='time',
    target_col='device1_value4',
    level=99.99,
)


# df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity-short-with-ex-vars.csv')

# df = pd.read_csv('https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/peyton_manning.csv')

# Add date features for anomaly detection
# Here, we use date features at the month and year levels
# anomalies_df_x = nixtla_client.detect_anomalies(
#     df, time_col='ds', 
#     target_col='y', 
#     freq='D',
#     level=99.99,
# )


# print(df.shape)
# print(df.columns)
# print(f"df: *** \n {df}")

# anomalies_df = nixtla_client.detect_anomalies(
#     df=df,
#     time_col='ds',
#     target_col='y'
# )

# # Plot weight of exgeonous features
# nixtla_client.weights_x.plot.barh(x='features', y='weights')