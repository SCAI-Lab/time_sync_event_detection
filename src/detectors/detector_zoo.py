import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm

from sklearn.pipeline import make_pipeline

from nixtla import NixtlaClient


def get_model(sensor_config):
    model_name = sensor_config['model_name']
    if model_name == 'IsolationForest':
        model = IsolationForest(contamination=0.0001, random_state=42)
    elif model_name == 'OneClassSVM':
        model = svm.OneClassSVM(nu=0.0001, kernel="rbf", gamma=0.1)
    elif model_name == 'LocalOutlierFactor':
        model = LocalOutlierFactor(n_neighbors=35, contamination=0.05)
    elif model_name == 'SGDOneClassSVM':
        model = make_pipeline(
            Nystroem(gamma=0.1, random_state=42, n_components=100),
            SGDOneClassSVM(
                nu=0.01,
                shuffle=True,
                fit_intercept=True,
                random_state=42,
                tol=1e-6,
            )
        )
    elif model_name == 'TimeGPT':
        nixtla_client = NixtlaClient(
            # defaults to os.environ.get("NIXTLA_API_KEY")
        )
        
        
        return nixtla_client
    else: return NotImplementedError

    return model
