import torchaudio
import torch.nn as nn
import torch
import torch.utils.data as data
import numpy as np
import glob
import os

from utils import all_seed
   
eps = 1e-6

# Data
class prep_dataset(torch.utils.data.Dataset):
    def __init__(self, ds_dir):
        self.ds_dir = ds_dir
        self.ds_files = [x for x in os.listdir(ds_dir) if "wav" in x]

    def __len__(self):
        return len(self.ds_files)

    def __getitem__(self, idx):
        # Select sample
        filepath = self.ds_files[idx]
        filename = "{}/{}".format(self.ds_dir, filepath)
        waveform, _ = torchaudio.load(filename)
        return waveform
    
class TIMIT_prep_dataset(torch.utils.data.Dataset):
    def __init__(self, ds_dir):
        self.ds_files = glob.glob('/home/kimsunw/TIMIT_Duan_data/Data/train/**/*.wav', recursive=True)

    def __len__(self):
        return len(self.ds_files)

    def __getitem__(self, idx):
        # Select sample
        filename = self.ds_files[idx]
        waveform, _ = torchaudio.load(filename)
        return waveform

def signal_preprocessing(waveform, n_frames, dtype):
    if dtype=='speech':
        if waveform.shape[1] <= n_frames:
            diff = n_frames - waveform.shape[1] + 1
            waveform = torch.nn.functional.pad(waveform, (0,diff))
    else:
        while waveform.shape[1] <= n_frames:
            waveform = torch.cat(2*[waveform],1)
    offset = np.random.randint(waveform.shape[1] - n_frames)
    waveform = waveform[:, offset:n_frames + offset]
    waveform = waveform / (waveform.std() + eps)
    return waveform

def data_processing(data, n_frames, dtype="speech", spkr_id_exempt=-1):
    list_waveforms = torch.zeros(len(data), n_frames)
    for idx, elem in enumerate(data):
        waveform = elem
        waveform = signal_preprocessing(waveform, n_frames, dtype)
        list_waveforms[idx] = waveform
    return list_waveforms

def standardize(seg):
    if seg.std() > 1e-3:
        seg = seg / (seg.std() + eps)
    return seg

def shuffle_set(x):
    r = torch.randperm(len(x))
    return x[r]

def mix_signals_batch(s, n, snr_ranges):
    """
    Checked.
    """
    n = scale_amplitude(n, snr_ranges)
    x = s + n
    
    # Standardize
    x = x/(x.std(1)[:,None] + eps)
    return x

# def prep_sig_ml(s,sr):
#     """
#     Checked. 
#     """
#     ml=np.minimum(s.shape[1], sr.shape[1])
#     s=s[:,:ml]
#     sr=sr[:,:ml]
#     return ml, s, sr

def get_mixing_snr(x, snr_ranges):
    snr_batch = np.random.uniform(
        low=min(snr_ranges), high=max(snr_ranges), size=len(x)).astype(np.float32)
    return torch.Tensor(snr_batch).to(x.device)

def scale_amplitude(x, snr_ranges):
    """
    Scale signal x by values within snr_ranges
    
    e: est_batch. Located on GPU.
    g: speech_batch/noise_batch. 
    """    
    # Compute mixing SNR
    snr_batch = get_mixing_snr(x, snr_ranges)
    x = x * (10 ** (-snr_batch / 20.))[:,None]
    return x

def apply_scale_invariance(s, x):
    """
    Checked. 
    """
    alpha = s.mm(x.T).diag()
    alpha /= ((s ** 2).sum(dim=1) + eps)
    alpha = alpha.unsqueeze(1)
    s = s * alpha
    return s