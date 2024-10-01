import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
import time
import pytz

import gzip
import shutil

MYTZ = "Europe/Zurich"
tzinfo = pytz.timezone(MYTZ)

def is_gz_file(filepath):
    # Check if the file has a .gz extension
    return filepath.endswith('.gz')


def unzip_gz_file(filepath, output_dir=None):
    if not is_gz_file(filepath):
        return ''
    
    # Define the output file path
    if output_dir is None:
        output_filepath = os.path.splitext(filepath)[0]  # Remove the .gz extension
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_filepath = os.path.join(output_dir, os.path.basename(os.path.splitext(filepath)[0]))

    # Unzip the .gz file
    with gzip.open(filepath, 'rb') as f_in:
        with open(output_filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    return output_filepath


def unzipper(path):
    """
    recursively unzip all the csv files in  subfolders under the directory
    """
    if not os.path.isdir(path):
        
        output_file = unzip_gz_file(path)
        return
    
    for entry in os.listdir(path):
        new_path = os.path.join(path, entry)
        unzipper(new_path)


def read_csv(path):
    """
    recursively read all the csv files in  subfolders under the directory
    """
    if not os.path.isdir(path):
        if path.endswith('.csv'):
            temp_df = pd.read_csv(path)
            return temp_df
        else: return
    
    combined_df = pd.DataFrame()
    for file in os.listdir(path):
        new_path = os.path.join(path, file)
        temp = read_csv(new_path)
        combined_df = pd.concat([combined_df, temp], axis=0, ignore_index=True)
    return combined_df