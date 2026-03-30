@git clone https://github.com/PKUDigitalHealth/ECGFounder.git
@conda create -n ECGFounder python=3.10
@conda activate ECGFounder
@pip install -r requirements.txt
@!mkdir checkpoint
@!wget https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/12_lead_ECGFounder.pth


import os
import glob
import xmltodict
import sys
sys.path.append('/opt/notebooks/ECGFounder')
from pathlib import Path
import random
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
# from sklearn.metrics import classification_report, confusion_matrix

from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from util import save_checkpoint, save_reg_checkpoint, my_eval_with_dynamic_thresh
from finetune_model import ft_12lead_ECGFounder
# from finetune_model import ft_12lead_ECGFounder, ft_1lead_ECGFounder
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wfdb
import gc
import time
import numpy as np
from scipy.signal import iirnotch, filtfilt, butter, medfilt

def z_score_normalization(signal):
    clean_signal = np.nan_to_num(signal, nan=np.nanmean(signal))
    if np.std(clean_signal) == 0:  # Handle zero variance
        return clean_signal - np.mean(clean_signal)
    return (clean_signal - np.mean(clean_signal)) / (np.std(clean_signal) +1e-8) 

def filter_bandpass(signal, fs):
    """
    Bandpass filter
    :param signal: 2D numpy array of shape (channels, time)
    :param fs: sampling frequency
    :return: filtered signal
    """
    # Remove power-line interference
    b, a = iirnotch(50, 30, fs)
    filtered_signal = np.zeros_like(signal)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, signal[c])

    # Simple bandpass filter
    b, a = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=fs)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, filtered_signal[c])

    # Remove baseline wander
    baseline = np.zeros_like(filtered_signal)
    for c in range(filtered_signal.shape[0]):
        kernel_size = int(0.4 * fs) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        baseline[c] = medfilt(filtered_signal[c], kernel_size=kernel_size)
    filter_ecg = filtered_signal - baseline

    return filter_ecg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
n_classes=2
pth = '/opt/notebooks/ECGFounder/checkpoint/12_lead_ECGFounder.pth'
model = ft_12lead_ECGFounder(device, pth, n_classes,linear_prob=False)

fs = 500  # set this to the actual sampling frequency of your ECG data

for item in s:
    huh = item.split('/')[-1].split('.')[0]
    try:
        with open(item) as f:
            data = xmltodict.parse(f.read())

        record = []
        strip_data = data['CardiologyXML']['StripData']

        # Case 1: WaveformData is directly under StripData
        if 'WaveformData' in strip_data:
            for wf in strip_data['WaveformData']:
                s_txt = str(wf['#text']).replace("\n", "").replace("\t", "")
                arr = np.array(list(map(int, s_txt.split(","))))
                record.append(arr)

        # Case 2: WaveformData is under StripData -> Strip
        elif 'Strip' in strip_data and 'WaveformData' in strip_data['Strip']:
            for wf in strip_data['Strip']['WaveformData']:
                s_txt = str(wf['#text']).replace("\n", "").replace("\t", "")
                arr = np.array(list(map(int, s_txt.split(","))))
                record.append(arr)

        # record: list of arrays (time,) for each lead
        record_arr = np.array(record)  # shape: (channels, time)

        # --- Apply your filter here ---
        filtered_arr = filter_bandpass(record_arr, fs=fs)

        # --- Then normalize (z-score per lead) ---
        nor_arr = np.array([z_score_normalization(row) for row in filtered_arr])

        # --- Prepare tensor for model ---
        signal_input = (
            torch.from_numpy(nor_arr)   # (channels, time)
            .float()
            .unsqueeze(0)               # (1, channels, time)
            .to(device)
        )

        with torch.no_grad():
            classification_logits, extracted_embeddings = model(signal_input)

        extracted_embeddings = extracted_embeddings.cpu().squeeze().detach().numpy()
        extracted_embeddings_df = pd.DataFrame(extracted_embeddings)

        file_path = f'./results_ecg/{huh}_ECGFounder.csv'
        if os.path.exists(file_path):
            print(f"File exists, rename: {file_path}")
            extracted_embeddings_df.to_csv(f"{file_path.split('.csv')[0]}_1.csv")
        else:
            extracted_embeddings_df.to_csv(file_path)

        os.system(f'dx upload --parents --path ./cache2/ {file_path}')

    except Exception as e:
        print(f"{huh}_bad")
        print(f"Error occurred: {e}")
        continue

extracted_embeddings_df.T
