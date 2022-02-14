import numpy as np
import torch
import mir_eval
import pesq
import torch_stoi
import pickle
import torch.nn as nn
import os
from datetime import datetime
import random
import logging

eps = 1e-6

### utils.py

def pt_to_np(X):
    return X.detach().cpu().numpy()

def get_stats(X):
    return print("Mean: {:.2f}, Std: {:.2f}, Min: {:.2f}, Max: {:.2f}".format(
        X.mean(), X.std(), X.min(), X.max()))

def save_model(model, output_directory, rundata, is_last=False):
    curr_epoch = rundata['epoch']
    if is_last:
        suffix = curr_epoch
    else:
        suffix = 'best'
    
    model_save_dir = "{}/Dmodel_{}.pt".format(output_directory, suffix)
    data_save_dir = "{}/rundata_{}.pt".format(output_directory, suffix)
    torch.save(model.state_dict(), model_save_dir)
    logging.info("D model saved to {}".format(model_save_dir))
    with open(data_save_dir, 'wb') as handle:
        pickle.dump(rundata, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(model, load_model):
    state_dict = torch.load(load_model, map_location=torch.device("cpu"))#map_location=lambda storage, loc: storage) 
    model.load_state_dict(state_dict)
    del state_dict
    logging.info("Loaded model from {}".format(load_model))

def load_rundata(load_rundata):
    with open(load_rundata, 'rb') as handle:
        rundata = pickle.load(handle)
    logging.info("Loaded rundata from {}".format(load_rundata))
    return rundata

def all_seed(seed):
    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

def setup_expr(args):
    args.n_frames = args.sr * args.duration
    args.stft_features = int(args.fft_size//2+1)
    args.stft_frames = int(np.ceil(args.n_frames/args.hop_size))+1

    t_stamp = '{0:%m%d%H%M}'.format(datetime.now())

    if args.rnn:
        tea_opt = "RNN_{}x{}".format(args.num_layers, args.hidden_size)
    else:
        tea_opt = "FC_{}x{}".format(args.num_layers, args.hidden_size)
    output_directory = "{}/{}/lr{:.0e}/betas_{}x{}/expr{}_ep{}_bs{}_nfrm{}_GPU{}".format(
        args.save_dir, 
        tea_opt,
        args.learning_rate,
        args.b1, args.b2,
        t_stamp,
        args.tot_epoch,
        args.batch_size, 
        args.n_frames, 
        args.device)

    print("Output Dir: {}".format(output_directory))
    if args.is_save:
        os.makedirs(output_directory, exist_ok=True)
        print("Created dir...")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [PID %(process)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(os.path.join(output_directory, "training.log")),
                logging.StreamHandler(),
            ],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [PID %(process)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    return output_directory

### dnn utils

eps = 1e-10 
stoi_loss = torch_stoi.NegSTOILoss(sample_rate=16000, extended=True)

def prep_sig_ml(s,sr):
    ml=np.minimum(len(s), len(sr))
    s=s[:ml]
    sr=sr[:ml]
    return ml, s, sr

def stft(signal, fft_size, hop_size):
    window = torch.hann_window(fft_size, device=signal.device)
    S = torch.stft(signal, n_fft=fft_size, hop_length=hop_size, window=window)#, return_complex=False)
    return S

def get_magnitude(S):
    S_mag = torch.sqrt(S[..., 0] ** 2 + S[..., 1] ** 2 + 1e-20)
    return S_mag

def apply_mask(spectrogram, mask, device):
    assert (spectrogram[...,0].shape == mask.shape)
    spectrogram2 = torch.zeros(spectrogram.shape)
    spectrogram2[..., 0] = spectrogram[..., 0] * mask
    spectrogram2[..., 1] = spectrogram[..., 1] * mask
    return spectrogram2.to(device)

def istft(spectrogram, fft_size, hop_size):
    window = torch.hann_window(fft_size, device=spectrogram.device)
    y = torch.istft(spectrogram, n_fft=fft_size, hop_length=hop_size, window=window)
    return y

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

def loss_sdr(source_signal, estimated_signal, offset=None):
    """
    Imported from: https://github.com/IU-SAIGE/sparse_mle
    """
    return -1.*torch.mean(calculate_sdr(source_signal, estimated_signal, offset))

def loss_sisdr(source_signal, estimated_signal, offset=None):
    """
    Imported from: https://github.com/IU-SAIGE/sparse_mle
    """
    return -1.*torch.mean(calculate_sisdr(source_signal, estimated_signal, offset))

def denoise_signal(args, mix_batch, G_model):
    """
    Return predicted clean speech.
    
    mix_batch and G_model: Located on GPU.
    """
    X = stft(mix_batch, args.fft_size, args.hop_size)
    X_mag = get_magnitude(X).permute(0,2,1)
    mask_pred = G_model(X_mag).permute(0,2,1)
    mask_pred = mask_pred.unsqueeze(3).repeat(1,1,1,2)
    X_est = X * mask_pred
    est_batch = istft(X_est, args.fft_size, args.hop_size)
    return est_batch

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

def get_scores(args, vas, van, model=None, is_ctn=False):
    
    N_va = len(vas)

    ml=np.zeros(N_va)
    mirSDRlist=np.zeros(N_va)
    SISNRlist=np.zeros(N_va)
    PESQlist=np.zeros(N_va)
    ESTOIlist=np.zeros(N_va)
    
    for j in range(N_va):
        
        s = vas[j]
        n = van[j]
        slen = s.shape[-1]
        nlen = n.shape[-1]
#         if slen > 160000:
#             offset = np.random.randint(slen - 160000)
#             s = s[:,offset:offset+160000]
#             slen = s.shape[-1]
#         while nlen <= slen:
#             n = np.concatenate([n]*2)
#             nlen = n.shape[-1]
#         offset = np.random.randint(nlen - slen)
#         n = n[offset:offset+slen]
        s /= (s.std() + eps)
        n /= (n.std() + eps)
        x = s+n
                    
        if model:
            speech_batch = torch.cuda.FloatTensor(s[None,:])
            noise_batch = torch.cuda.FloatTensor(n[None,:])
            mix_batch = torch.cuda.FloatTensor(x[None,:])
            
            if is_ctn:
                est_batch = model(mix_batch).squeeze(1)
            else:
                S = stft(speech_batch, args.fft_size, args.hop_size)
                N = stft(noise_batch, args.fft_size, args.hop_size)
                X = stft(mix_batch, args.fft_size, args.hop_size)
                S_mag = get_magnitude(S)
                N_mag = get_magnitude(N)
                if model=="IBM":
                    mask_pred = (S_mag > N_mag) * 1.0
                else:
                    X_mag = get_magnitude(X)
                    X_mag = X_mag.permute(0,2,1)
                    mask_pred = model(X_mag.to(args.device))
                    mask_pred = mask_pred.permute(0,2,1)
                mask_pred = mask_pred.unsqueeze(3).repeat(1,1,1,2)
                X_est = X.to(args.device) * mask_pred
                est_batch = istft(X_est, args.fft_size, args.hop_size)
            recon = est_batch.detach().cpu().numpy()[0]
        else:
            recon = x + 1e-10
        
        ml_i, ref, enh = prep_sig_ml(s, recon)

        msdr, msir, msar = get_mir_scores(s, n, x, recon)
        ml[j] = ml_i
        mirSDRlist[j] = msdr
        # SISNR
        SISNRlist[j] = calculate_sisdr(torch.Tensor(s), torch.Tensor(recon))
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
            
    curr_tot_mSDR = np.sum(ml*mirSDRlist/np.sum(ml))                              
    curr_tot_SISNR = np.sum(ml*SISNRlist/np.sum(ml))
    curr_tot_PESQ = np.sum(ml*PESQlist/np.sum(ml))
    curr_tot_ESTOI = np.sum(ml*ESTOIlist/np.sum(ml))
    
    return curr_tot_mSDR, curr_tot_SISNR, curr_tot_PESQ, curr_tot_ESTOI
            
            
def run_se(args, SE_model, speech_dataloader, noise_dataloader, is_train=True, optimizer=None):
        total_loss = []
        noise_iter = iter(noise_dataloader)
        for batch_idx, speech_batch in enumerate(speech_dataloader):
            try:
                noise_batch = next(noise_iter)
            except StopIteration:
                noise_iter = iter(noise_dataloader)
                noise_batch = next(noise_iter)
                
            speech_batch = speech_batch.to(torch.device("cuda"))
            noise_batch = noise_batch.to(torch.device("cuda"))
            mix_batch = speech_batch + noise_batch

            S = stft(speech_batch, args.fft_size, args.hop_size)
            N = stft(noise_batch, args.fft_size, args.hop_size)
            X = stft(mix_batch, args.fft_size, args.hop_size)
            S_mag = get_magnitude(S)
            N_mag = get_magnitude(N)
            X_mag = get_magnitude(X)

            IBM = (S_mag > N_mag) * 1.0        
    #         IBM = S_mag / (S_mag + N_mag)

            X_mag = X_mag.permute(0,2,1)
            mask_pred = SE_model(X_mag.to(args.device))
            mask_pred = mask_pred.permute(0,2,1)

    #         loss_i = loss_fn(IBM, mask_pred)
            mask_pred = mask_pred.unsqueeze(3).repeat(1,1,1,2)
            X_est = X.to(args.device) * mask_pred
            est_batch = istft(X_est, args.fft_size, args.hop_size)
            loss_i = loss_sisdr(speech_batch, est_batch)
            
            if is_train:
                optimizer.zero_grad()
                loss_i.backward()
                optimizer.step()

    #         total_loss.append(loss_i.mean().detach().cpu())
            total_loss.append(float(loss_i))
        
#             if (batch_idx % args.print_every) == 0: 
#                 print ("Batch {}. Loss: {:.3f} SI-SDRi".format(
#                         batch_idx, np.mean(total_loss))) # logging.info
        
        return np.mean(total_loss)
    