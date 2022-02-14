from argparse import ArgumentParser
import numpy as np
import librosa
import pickle
import os
import torch

from loader import *
from utils import save_pkl, load_pkl

def parse_arguments():
    parser = ArgumentParser()    
    parser.add_argument("--seed", type=int, default=0,
                        help = "Seed for train and test speaker selection")
    
    parser.add_argument("--use_stft", action='store_true', help="Use STFT features")
    parser.add_argument("--use_rbf_target", action='store_true', help="Use RBF Kernels to generate SSMs")
    parser.add_argument("--gen_ssm", action='store_true', help="Generate SSM dataloader")
    parser.add_argument("--gen_sgbs", action='store_true', help="Generate feature dataloader")

    parser.add_argument("--sigma2", type=float, default=0.0, help="Kernel width parameter")
    parser.add_argument("--segment_len", type=int, default=1000, help="Number of frames to include per SSM")
    parser.add_argument("--gpu", type=int, default=0, help="GPU Card id")
    
    parser.add_argument("--test", action='store_true', help="For debugging")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    np.random.seed(args.seed)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    # Load and shuffle data
    if args.use_stft:
        Xtr_load_nm = "Xtr_STFT.npy"
    else:
        Xtr_load_nm = "Xtr_Mel.npy"
    Xtr = np.load(Xtr_load_nm)
    print("Loaded data")
    truncate_len = len(Xtr) % args.segment_len
    np.random.seed(args.seed)
    shuffle_idx = np.random.permutation(len(Xtr))
    Xtr_shuffled = Xtr[shuffle_idx][:-truncate_len]
    Ntr, n_features = Xtr_shuffled.shape

    # Create directories for SSM and feature dataloaders
    if args.use_stft:
        prefix = "STFT_"
    else:
        prefix = "Mel_"
    
    sgbs_dir = "{}Sgbs".format(prefix)
    if args.use_rbf_target:
        prefix += "RBF{}_".format(str(args.sigma2))
    ssm_dir = "{}SSM".format(prefix)
    if not args.test:
        os.makedirs(ssm_dir, exist_ok=True)
        os.makedirs(sgbs_dir, exist_ok=True)

    # Generate files for dataloader
    for i in range(0, Ntr, args.segment_len):
        seg_idx = i//args.segment_len
        X_seg = torch.cuda.FloatTensor(Xtr_shuffled[i:i+args.segment_len])
        if args.use_rbf_target:
            ssm = torch.exp(torch.mm(X_seg, X_seg.t()) / args.sigma2)/np.exp(1/args.sigma2)
        else:
            ssm = torch.mm(X_seg, X_seg.t())
        
        X_seg = X_seg.detach().cpu().numpy()    
        ssm = ssm.detach().cpu().numpy()
    
        savenm = "{}/{}_{}".format(ssm_dir, ssm_dir, seg_idx)
        print("Created {}".format(savenm))
        if not args.test:
            os.makedirs(savenm, exist_ok=True)
            np.save("{}/ssm".format(savenm), ssm)
        
        savenm = "{}/{}_{}".format(sgbs_dir, sgbs_dir, seg_idx)
        print("Created {}".format(savenm))
        X_seg_bias = np.concatenate((X_seg, np.ones((len(X_seg),1))),1).astype(np.float32)
        if not args.test:
            os.makedirs(savenm, exist_ok=True)
            np.save("{}/sgbs".format(savenm), X_seg_bias)
            
if __name__ == "__main__":
    main()
    

#     python generate_dataloaders.py --use_stft --use_rbf_target --sigma2 0.9 --gpu 3 --test