import numpy as np
import torch
import mir_eval
import pesq
import torch_stoi

eps = 1e-10 
stoi_loss = torch_stoi.NegSTOILoss(sample_rate=16000, extended=True)

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

def get_mir_scores(s, n, x, sr):
    """
    s: Source signal
    n: Noise signal
    x: s + n
    sr: Reconstructed signal (or some signal to evaluate against)
    """
    ml = np.int(np.minimum(len(s), len(sr)))
    source = np.array(s[:ml])[:,None].T
    noise = np.array(n[:ml])[:,None].T
    sourceR = np.array(sr[:ml])[:,None].T
    noiseR = np.array(x[:ml]-sr[:ml])[:,None].T
    sdr,sir,sar,_=mir_eval.separation.bss_eval_sources(
            np.concatenate((source, noise),0),
            np.concatenate((sourceR, noiseR),0), 
            compute_permutation=False)   
    # Take the first element from list for source's performance
    return sdr[0],sir[0],sar[0]

def get_scores(val_waves_dict, recons):
    vas = val_waves_dict['s']
    van = val_waves_dict['n']
    vax = val_waves_dict['x']

    N_va = len(vax)

    ml=np.zeros(N_va)
    mirSDRlist=np.zeros(N_va)
    SISNRlist=np.zeros(N_va)
    PESQlist=np.zeros(N_va)
    ESTOIlist=np.zeros(N_va)
    
    for j in range(N_va):
        recon = recons[j]
        ml_i, ref, enh = prep_sig_ml(vas[j], recon)
        # SDR
        msdr, msir, msar = get_mir_scores(vas[j], van[j], vax[j], recon)
        ml[j] = ml_i
        mirSDRlist[j] = msdr
        # SISNR
        SISNRlist[j] = calculate_sisdr(torch.Tensor(vas[j]), torch.Tensor(recon))
        # PESQ
        PESQlist[j] = pesq.pesq(16000, ref, enh, 'wb')
        # ESTOI
        ESTOIlist[j] = -float(stoi_loss(torch.Tensor(enh), torch.Tensor(ref)))

        # Print intermediate results
        if (j+1) % 20 == 0:
            prog = (j+1)/len(vas)*100
            print_line = "{}: {:.2f}% | ".format(j+1, prog)
            # SDR
            curr_tot_mSDR = np.sum(ml*mirSDRlist/np.sum(ml))                              
            print_line += "mSDR {:.3f} | ".format(curr_tot_mSDR)
            # SISNR
            curr_tot_SISNR = np.sum(ml*SISNRlist/np.sum(ml))
            print_line += "SISNR {:.3f} | ".format(curr_tot_SISNR)
            # PESQ
            curr_tot_PESQ = np.sum(ml*PESQlist/np.sum(ml))
            print_line += "PESQ {:.3f} | ".format(curr_tot_PESQ)
            # ESTOI
            curr_tot_ESTOI = np.sum(ml*ESTOIlist/np.sum(ml))
            print_line += "ESTOI {:.4f}".format(curr_tot_ESTOI)
            print (print_line)