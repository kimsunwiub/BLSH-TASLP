import numpy as np
import mir_eval
import pickle
import torch
import random
from torchvision import datasets

# Misc. helper functions
def count_mins(X):
    tot_mins = 0
    for i in range(len(X)):
        tot_mins += len(X[i])
    tot_mins /= (16000 * 60)
    print ("{} mins".format(int(np.round(tot_mins))))
    
def save_pkl(X, name):
    with open(name, 'wb') as handle: 
        pickle.dump({'temp': X}, handle)
    return

def load_pkl(name):
    with open(name, 'rb') as handle: 
        X = pickle.load(handle)['temp']
    return X

def SDR(s,sr):
    eps=1e-20
    ml=np.minimum(len(s), len(sr))
    s=s[:ml]
    sr=sr[:ml]
    return ml, 10*np.log10(np.sum(s**2)/(np.sum((s-sr)**2)+eps)+eps)

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

def pt_to_np(X):
    return X.detach().cpu().numpy()

def bssm_tanh(input_data, p_m):
    # For debugging purposes
    temp = torch.mm(input_data, p_m)
    output = torch.tanh(temp)
    bssm = torch.mm(output, output.t())
    bssm = (bssm+1)/2
    return bssm

# Helper functions for training weak learners
class signBNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x+1e-10).sign()
    @staticmethod
    def backward(ctx, grad_y):
        x, = ctx.saved_tensors
        return grad_y.mul(1-torch.tanh(x)**2)

def bssm_sign(input_data, p_m, is_train=True): #TODO BIAS
    temp = torch.mm(input_data, p_m) # 1000 x 513. dot 513 x 1
    if is_train:
        output = signBNN.apply(temp) # 1000 x 1
    else:
        output = torch.sign(temp)
    bssm = torch.mm(output, output.t()) # 1000 x 1000
    return bssm

def xent_fn(p, q):
    ret_xent = -(
                p*torch.log(q+1e-10) \
                + (1-p) * torch.log(1-q+1e-10)
            )
    return ret_xent

def np_xent_fn(p, q):
    ret_xent = -(
                (p)*np.log(q+1e-20) \
                + (1-p) * np.log(1-q+1e-20)
            )
    return ret_xent

def validate(Xva, args, p_m):
    # Validation
    epoch_losses = []
    for i in range(0, len(Xva), args.segment_len):
        Xva_seg = torch.cuda.FloatTensor(Xva[i:i+args.segment_len])
        ssm_va = torch.mm(Xva_seg, Xva_seg.t())
        ssm_va /= ssm_va.max()
        Xva_seg_bias = torch.cat((Xva_seg, torch.ones((len(Xva_seg),1)).cuda()), 1)
        bssm = bssm_sign(Xva_seg_bias, p_m, False)
        if args.lossfn == 'xent':
            bssm = (bssm+1)/2
            print("DEBUG")
        loss_va = ((bssm-ssm_va)**2).mean()
        epoch_losses.append(float(loss_va))
    return np.mean(epoch_losses)

def get_stats(X):
    return X.mean(), X.std(), X.min(), X.max()

def get_beta_r2(Xtr_ssm_dl, Xtr_sgbs_dl, wi_dir, p_m, args):
    e_t_list = []
    for i, data in enumerate(zip(Xtr_ssm_dl, Xtr_sgbs_dl)):
        ssm_tr = data[0][0][0].cuda()
        Xtr_seg_bias = data[1][0][0].cuda()
        wi_seg = np.load("{}/w_{}.npy".format(wi_dir, i), allow_pickle=True)
        bssm = bssm_sign(Xtr_seg_bias, p_m, is_train=False)
        bssm = (bssm+1)/2
        
        # Backprop with weighted sum of errors
        if args.lossfn == 'xent':
            L = xent_fn(bssm,ssm_tr)
        else:
            L = torch.abs(bssm-ssm_tr)
        D = L.max()

        if args.lossfn == 'xent' or args.lossfn == 'linear':
            err = L/D
        elif args.lossfn == 'square':
            err = L**2/D**2
        elif args.lossfn == 'exponential':
            err = 1-torch.exp(-torch.abs(L)/D)

        err = pt_to_np(err)
        e_t = (err * wi_seg).sum()
        
        e_t_list.append(e_t)
    e_t_mean = np.mean(e_t_list)+1e-10
    assert e_t_mean < 1.0
    beta = e_t_mean/(1-e_t_mean)

    return beta

def get_beta(Xtr_ssm_dl, Xtr_sgbs_dl, wi_dir, p_m, args):
    e_t_list = []
    for i, data in enumerate(zip(Xtr_ssm_dl, Xtr_sgbs_dl)):
        ssm = data[0][0][0].cuda()
        Xtr_seg_bias = data[1][0][0].cuda()
        wi_seg = np.load("{}/w_{}.npy".format(wi_dir, i), allow_pickle=True)
        bssm = bssm_sign(Xtr_seg_bias, p_m, is_train=False)
        
        if args.ada_target_scale_zero:
            # Scale BSSM [-1,1] -> [0,1]
            bssm = (bssm+1)/2
        else:
            # Scale SSM [0,1] -> [-1,1]
            ssm = ssm*2-1
            
        err = bssm-ssm
        if not args.ada_target_scale_zero:
            err = err/2.
        err = err**2
            
        err = pt_to_np(err)
        e_t = (err * wi_seg).sum()
        e_t_list.append(e_t)

    e_t_mean = np.mean(e_t_list) + 1e-10 
    assert e_t_mean < 1.0
    
    if args.ada_target_scale_zero:
        beta = np.log((1-e_t_mean)/e_t_mean)
    else:
        beta = 0.5 * np.log((1-e_t_mean)/e_t_mean)
    
    return beta



def all_seed(seed):
    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

def load_ssm_dataloader(args):
    if args.use_stft_target:
        root = "STFT"
    elif args.use_logMel_target:
        root = "logMel"
    elif args.use_mel_target:
        root = "Mel"
    if args.use_rbf_target:
        root += "_RBF{}".format(args.sigma2)
    root += "_SSM"
    ssm_dataset = datasets.DatasetFolder(
        root=root,
        loader=npy_loader,
        extensions='.npy')
    kwargs = {'num_workers': 0, 'pin_memory': True}
    ssm_dataloader = torch.utils.data.DataLoader(
        ssm_dataset,
        batch_size=1, 
        shuffle=False, 
        **kwargs)
    print("Loaded ", root)
    return ssm_dataloader

def load_sgbs_dataloader(args):
    if args.use_stft_learner:
        root = "STFT_Sgbs"
    elif args.use_logMel_learner:
        root = "logMel_Sgbs"
    elif args.use_mel_learner:
        root = "Mel_Sgbs"
    sgbs_dataset = datasets.DatasetFolder(
        root=root,
        loader=npy_loader,
        extensions='.npy')
    kwargs = {'num_workers': 0, 'pin_memory': True}
    sgbs_dataloader = torch.utils.data.DataLoader(
        sgbs_dataset,
        batch_size=1, 
        shuffle=False, 
        **kwargs)
    print("Loaded ", root)
    return sgbs_dataloader

"""
        # Create SSM with a kernel
        if args.kernel == "linear":
            ssm_tr = torch.mm(Xtr_seg, Xtr_seg.t())
        elif args.kernel == "rbf":
            ssm_tr = torch.exp(torch.mm(Xtr_seg, Xtr_seg.t()) / args.sigma2)
            
            
        
"""