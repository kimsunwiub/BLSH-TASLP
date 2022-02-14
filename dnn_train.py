import sys
import pickle
import torchaudio
import torch.nn as nn
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
from time import time
from datetime import datetime
from argparse import ArgumentParser
import logging
import random
from torch.autograd import grad
from torch.autograd import Variable

import torch
torch.set_num_threads(1)

import torch.jit as jit
import torch.nn as nn
from torch.nn import Parameter
from typing import List, Tuple
from torch import Tensor
import math

from utils import *
from data import *
from models import *

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-e", "--tot_epoch", type=int, default=200)
    parser.add_argument("-l", "--num_layers", type=int, default=-1)
    parser.add_argument("-u", "--hidden_size", type=int, default=-1)

    parser.add_argument("--save_dir", type=str, default="/home/kimsunw/workspace/blsh_dnn/bn_models_results/")    
#     parser.add_argument("--save_dir", type=str, default="/home/kimsunw/workspace/blsh_dnn/models_results/")    
    parser.add_argument("--load_SEmodel", type=str, default=None) # "models_results/expr11101813_SE_G2x1024_D-1x-1Op_lr1e-04_bs100_dr0.0_GPU2/model_199.pt"
    parser.add_argument("--load_SErundata", type=str, default=None)
    
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--print_every", type=int, default=1500)
    parser.add_argument("--validate_every", type=int, default=2)
    
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--b1", type=float, default=.9)
    parser.add_argument("--b2", type=float, default=.999)
    parser.add_argument("--fft_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=256)
    
    parser.add_argument('--is_save', action='store_true')
    
    parser.add_argument('--rnn', action='store_true')
    parser.add_argument('--fc', action='store_true')
    
    parser.add_argument("--data_dir", type=str, default="/home/kimsunw/data/transfer_data/")
    parser.add_argument("--snr_ranges", nargs='+', type=int, default=[-5,10])   
        
    return parser.parse_args()

args = parse_arguments()
output_directory = setup_expr(args)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)
args.device = 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if args.rnn:
    SE_model = RNN_model(args.hidden_size, args.num_layers, args.stft_features)
elif args.fc:
    if args.num_layers == 3:
        SE_model = FC3_model(args.hidden_size, args.stft_features)
    else:
        SE_model = FC2_model(args.hidden_size, args.stft_features)

if args.load_SEmodel:
    load_model(SE_model, args.load_SEmodel)
    
print(SE_model)
SE_model = SE_model.to(args.device)
optimizer = torch.optim.Adam(params=SE_model.parameters(),lr=args.learning_rate, betas=(args.b1,args.b2))
# loss_fn = nn.MSELoss()

### Data
eps = args.eps

tr_speech_ds = prep_dataset('data/train/speech/')
# tr_speech_ds = TIMIT_prep_dataset()
va_speech_ds = prep_dataset('data/val/speech/')
# va_speech_ds = prep_dataset('data/test/speech/')
tr_noise_ds = prep_dataset('data/train/noise/')
va_noise_ds = prep_dataset('data/val/noise/')
# va_noise_ds = prep_dataset('data/test/noise/')

kwargs = {'num_workers': 0, 'pin_memory': True, 'drop_last': True}

tr_speech_dataloader = data.DataLoader(dataset=tr_speech_ds,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn= lambda x: data_processing(x, args.n_frames, "speech"),
    **kwargs)
va_speech_dataloader = data.DataLoader(dataset=va_speech_ds,
    batch_size=1,
    shuffle=False,
    collate_fn= lambda x: data_processing(x, args.n_frames, "speech"),
    **kwargs)

tr_noise_dataloader = data.DataLoader(dataset=tr_noise_ds,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn= lambda x: data_processing(x, args.n_frames, "noise"),
    **kwargs)
va_noise_dataloader = data.DataLoader(dataset=va_noise_ds,
    batch_size=1,
    shuffle=False,
    collate_fn= lambda x: data_processing(x, args.n_frames, "noise"),
    **kwargs)

tot_toc = time()
best_impr = 0
all_losses = []
all_sisdrs = []
for epoch in range(args.tot_epoch):      
    SE_model.train()
    tr_loss = run_se(args, SE_model, tr_speech_dataloader, tr_noise_dataloader, is_train=True, optimizer=optimizer)
    all_losses.append(tr_loss)
            
    SE_model.eval()
    va_loss = run_se(args, SE_model, va_speech_dataloader, va_noise_dataloader, is_train=False)   
    all_sisdrs.append(va_loss)
    
    print ("Epoch {}. Loss: {:.3f} SI-SDR".format(
            epoch, all_sisdrs[-1])) # logging.info
    rundata = {"epoch": epoch, "sisdr": best_impr, 
               "tr_losses": all_losses, "va_losses": all_sisdrs}   
    curr_impr = all_sisdrs[-1]
    
    if args.is_save:
        if best_impr > curr_impr:
            best_impr = curr_impr
            logging.info("Saved at %d" % epoch)
            save_model(SE_model, output_directory, rundata)

logging.info("Finished training SE")
rundata = {"epoch": epoch, "sisdr": best_impr, 
            "tr_losses": all_losses, "va_losses": all_sisdrs}    
if args.is_save:
    save_model(SE_model, output_directory, rundata, is_last=True)
