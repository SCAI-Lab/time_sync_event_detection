import pandas as pd
import numpy as np

def get_detector(model_name : str):
    if model_name == "TimeGPT":
        return detector_timeGPT
    else: return detector_sk



def detector_sk(window, model):
    """
    detector for using sklearn model
    """
    # print(f" *** window: \n {window}")
    features = pd.DataFrame(list(window))
    features = features.drop(columns=['time', 'timestamp'], errors='ignore')
    # model.fit(features)
    # latest_data = features[-1].reshape(1, -1)
    # score = model.decision_function(latest_data)[0]
    anomaly_scores = model.fit_predict(features)
    anomaly_scores = (anomaly_scores == -1).astype(int)
    
    return anomaly_scores

def detector_timeGPT(window, model):
    """
    detector for using TimeGPT
    """
    from nixtla import NixtlaClient
    nixtla_client = NixtlaClient()

    features = pd.DataFrame(list(window))
    anomalies_df = nixtla_client.detect_anomalies(
        features, 
        time_col='time', 
        target_col='value', 
        freq='D',
    )
    # model.fit(features)
    # latest_data = features[-1].reshape(1, -1)
    # score = model.decision_function(latest_data)[0]
    anomaly_scores = model.fit_predict(features)
    anomaly_scores = anomaly_scores == -1
    return anomaly_scores

def detector_own_model(window, model):
    return NotImplementedError 