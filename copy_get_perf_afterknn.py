from argparse import ArgumentParser
import numpy as np
import librosa
import pickle
import torch
torch.set_num_threads(1)

import os

from utils import SDR, get_mir_scores, load_pkl

import pesq
import torch_stoi

def parse_arguments():
    parser = ArgumentParser() 
    
    parser.add_argument("--sdr", action='store_true',
                        help="SDR")   
    parser.add_argument("--sisnr", action='store_true',
                        help="SDR")   
    parser.add_argument("--pesq", action='store_true',
                        help="SDR")   
    parser.add_argument("--estoi", action='store_true',
                        help="SDR")   
    
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Results Dir")    
    
    parser.add_argument("-n", "--n_proj", type=int, default=None,
                        help = "Number of projections")
    parser.add_argument("-p", "--use_perc", type=int, default=10,
                        help = "Random sample %% of training set")
    parser.add_argument("-d", "--seed", type=int, default=42,
                        help = "Seed for random sampling from dictionary")
    parser.add_argument("-e", "--print_every", type=int, default=500,
                        help = "Option for printing frequency")
    
    return parser.parse_args()

def prep_sig_ml(s,sr):
    ml=np.minimum(len(s), len(sr))
    s=s[:ml]
    sr=sr[:ml]
    return ml, s, sr

def calculate_sdr(source_signal, estimated_signal, offset=None, scale_invariant=False):
    s = source_signal.clone()
    y = estimated_signal.clone()

    # add a batch axis if non-existant
    if len(s.shape) != 2:
        s = s.unsqueeze(0)
        y = y.unsqueeze(0)

    # truncate all signals in the batch to match the minimum-length
    min_length = min(s.shape[-1], y.shape[-1])
    s = s[..., :min_length]
    y = y[..., :min_length]

    if scale_invariant:
        alpha = s.mm(y.T).diag()
        alpha /= ((s ** 2).sum(dim=1) + eps)
        alpha = alpha.unsqueeze(1)  # to allow broadcasting
    else:
        alpha = 1

    e_target = s * alpha
    e_res = e_target - y

    numerator = (e_target ** 2).sum(dim=1)
    denominator = (e_res ** 2).sum(dim=1) + eps
    sdr = 10 * torch.log10((numerator / denominator) + eps)

    # if `offset` is non-zero, this function returns the relative SDR
    # improvement for each signal in the batch
    if offset is not None:
        sdr -= offset

    return sdr

def calculate_sisdr(source_signal, estimated_signal, offset=None):
    return calculate_sdr(source_signal, estimated_signal, offset, True)

args = parse_arguments()

save_nm = "perc_scores/{}/Nprj[{}]_Perc[{}]_See[{}]".format(
            args.results_dir.split('Saved/')[1],
            args.n_proj, args.use_perc, args.seed)
os.makedirs(save_nm, exist_ok=True)

print_line = "Job: {}\n".format(save_nm)
print (print_line)

if args.estoi:
    stoi_loss = torch_stoi.NegSTOILoss(sample_rate=16000, extended=True)

with open('tesnx_wavefiles.pkl', 'rb') as handle:
    val_waves_dict = pickle.load(handle)

vas = val_waves_dict['s']
van = val_waves_dict['n']
vax = val_waves_dict['x']

results_dir = args.results_dir
sub_results = os.listdir(results_dir)

N_va = len(vas)
ml=np.zeros(N_va)
mirSDRlist=np.zeros(N_va)
SISNRlist=np.zeros(N_va)
PESQlist=np.zeros(N_va)
ESTOIlist=np.zeros(N_va)

for i in range(len(sub_results)):
    res_i = sub_results[i]

    proj_info = res_i.split('Prj[')[1].split(']_Fom')[0]    
    is_proj = bool(proj_info.split('|')[0])
    n_i = proj_info.split('|')[1]
    if n_i != "None":
        n_i = int(n_i)
    else:
        n_i = None
    
    perc_i = int(res_i.split('Per[')[1].split(']_Cld')[0])
    seed_i = int(res_i.split('See[')[1].split(']_Per')[0])
    
    if is_proj and args.n_proj == n_i and args.use_perc == perc_i and args.seed == seed_i:
        print(res_i)
        for j in range(N_va):
            recon = np.load("{}/recon_{}.npy".format(results_dir + '/' + res_i, j))
            ml_i, ref, enh = prep_sig_ml(vas[j], recon)
            if args.sdr:
                msdr, msir, msar = get_mir_scores(
                    vas[j], van[j], vax[j], recon)
                ml[j] = ml_i
                mirSDRlist[j] = msdr
            if args.sisnr:
                SISNRlist[j] = calculate_sisdr(torch.Tensor(vas[j]), torch.Tensor(recon))
            if args.pesq:
                PESQlist[j] = pesq.pesq(16000, ref, enh, 'wb')
            if args.estoi:
                ESTOIlist[j] = -float(stoi_loss(torch.Tensor(enh), torch.Tensor(ref)))

            # Print intermediate results
            if (j+1) % args.print_every == 0:
                prog = (j+1)/len(vas)*100
                print_line = "{}: {:.2f}% | ".format(j+1, prog)
                if args.sdr:
                    curr_tot_mSDR = np.sum(ml*mirSDRlist/np.sum(ml))                              
                    print_line += "mSDR {:.3f} | ".format(curr_tot_mSDR)
                if args.sisnr:
                    curr_tot_SISNR = np.sum(ml*SISNRlist/np.sum(ml))
                    print_line += "SISNR {:.3f} | ".format(curr_tot_SISNR)
                if args.pesq:
                    curr_tot_PESQ = np.sum(ml*PESQlist/np.sum(ml))
                    print_line += "PESQ {:.3f} | ".format(curr_tot_PESQ)
                if args.estoi:                
                    curr_tot_ESTOI = np.sum(ml*ESTOIlist/np.sum(ml))
                    print_line += "ESTOI {:.3f}".format(curr_tot_ESTOI)
                print (print_line)
                
print_line = "END | "
if args.sdr:
    curr_tot_mSDR = np.sum(ml*mirSDRlist/np.sum(ml))
    print_line += "mSDR {:.3f} | ".format(curr_tot_mSDR)
if args.sisnr:
    curr_tot_SISNR = np.sum(ml*SISNRlist/np.sum(ml))
    print_line += "SISNR {:.3f} | ".format(curr_tot_SISNR)
if args.pesq:
    curr_tot_PESQ = np.sum(ml*PESQlist/np.sum(ml))
    print_line += "PESQ {:.3f} | ".format(curr_tot_PESQ)
if args.estoi:                
    curr_tot_ESTOI = np.sum(ml*ESTOIlist/np.sum(ml))
    print_line += "ESTOI {:.3f} | ".format(curr_tot_ESTOI)
print (print_line)