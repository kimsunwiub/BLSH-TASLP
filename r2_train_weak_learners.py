from argparse import ArgumentParser
import numpy as np
import pickle
import time
import os
from datetime import datetime

import torch
torch.set_num_threads(1)

import torch.nn as nn
from torch.autograd import Variable

from utils import *

import warnings # TODO: Replace deprecated fn.
warnings.filterwarnings("ignore")

def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("--use_stft_learner", action='store_true', 
                        help="Using STFT features. Otherwise, log mel features are used.")
    parser.add_argument("--use_stft_target", action='store_true', 
                        help="Using STFT target. Otherwise, log mel targets are used.")
    parser.add_argument("--use_rbf_target", action='store_true', 
                        help="Using RBF kernel SSM target.")
    
    parser.add_argument("--gpu_id", type=int, default=0,
                        help = "GPU_ID")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Load previous permutations and results")
    parser.add_argument("--save_model", type=str, default=None,
                        help="Directory for saving")
    parser.add_argument("--min_iter", type=int, default=20,
                        help = "Minimum number of iterations for learners")
    parser.add_argument("--max_iter", type=int, default=200,
                        help = "Maximum number of iterations for learners")
    
    parser.add_argument("--seed", type=int, default=42,
                        help = "Data: Seed for speaker selection")
    parser.add_argument("--segment_len", type=int, default=1000,
                        help = "Segment length: ")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate for training weak learners")
    parser.add_argument("--num_proj", type=int, default=300,
                        help = "Number of projections")
    parser.add_argument("--kernel", type=str, default='linear', 
                        help="Kernel for SSM. Options: linear, rbf")
    parser.add_argument("--sigma2", type=float, default=0.0, 
                        help="Denominator value for RBF kernel")
    parser.add_argument("--save_every", type=int, default=25,
                        help = "Specify saving frequency")
    parser.add_argument("--debug", action='store_true',
                        help="Debugging option (tweak lr, save wip1s)")
    
    parser.add_argument("--lossfn", type=str, default='xent',
                        help="Default: 'xent'. Other: 'linear', 'square', or 'exponential'.")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
    
    # Load dataloaders
    Xtr_ssm_dl = load_ssm_dataloader(args.use_stft_target, args)
    Xtr_sgbs_dl = load_sgbs_dataloader(args.use_stft_learner, args)
 
    n_features = 513 if args.use_stft_learner else 128
    Ntr = len(Xtr_ssm_dl)
    feat = "STFT" if args.use_stft_learner else "LM"
    feat += "_"
    feat += "STFT" if args.use_stft_target else "LM"
    
    all_seed(args.seed)
    
    # Load previous model if given
    if args.load_model:
        output_directory = args.load_model
        projections = list(np.load("{}/projs.npy".format(output_directory)))
        proj_losses = load_pkl("{}/projlosses.pkl".format(output_directory))
        betas = list(np.load("{}/betas.npy".format(output_directory)))
        w1s = list(np.load("{}/w1s.npy".format(output_directory)))
        wi_dir = '{}/weights'.format(output_directory)
        m_start = len(projections)
    else:
        # Create root directory for experiment
        t_stamp = '{0:%m%d%H%M}'.format(datetime.now())
        model_nm = "feat({})_kern({}_{})_lr({:.0e})_loss({})".format(feat, args.kernel, args.sigma2, 
                                                                     args.lr, args.lossfn)
        output_directory = "{}/expr{}_{}_GPU{}".format(args.save_model, t_stamp, model_nm, args.gpu_id)
        print("Output Dir: {}".format(output_directory))
        os.makedirs(output_directory, exist_ok=True)
        print("Created dir...")
        wi_dir = '{}/weights'.format(output_directory)
        os.makedirs(wi_dir, exist_ok=True)
        
        # Initialize and save observation weights
        for i in range(Ntr):
            wi = np.ones((args.segment_len, args.segment_len))
            init_val = 1.0/(args.segment_len**2)
            wi *= init_val
            np.save("{}/w_{}".format(wi_dir,i), wi, allow_pickle=True)
        print("Created weights...")
        
        m_start = 0
        projections = []
        proj_losses = []
        betas = []
        w1s = []
    
    # Training parameters
    tol = 1/(args.segment_len**2) # Tolerance level for determining convergence
    momentum_range = 10 # Compute momentum for past momentum_range number of epochs
    for m in range(m_start, args.num_proj + m_start):
        # Init Training
        p_m = torch.Tensor(n_features+1, 1)
        p_m = Variable(
            torch.nn.init.xavier_normal_(p_m).cuda(), requires_grad=True)
        optimizer = torch.optim.Adam([p_m], betas=[0.95, 0.98], lr=args.lr)

        toc = time.time()
        epoch = 0
        tr_losses = []
        curr_momentum = np.inf
        diff = np.inf
        # Iterate until convergence or max_iter is reached
        while epoch < args.min_iter or (diff > tol and epoch < args.max_iter):
            ep_losses = []
            # Training
            for i, data in enumerate(zip(Xtr_ssm_dl, Xtr_sgbs_dl)):
                ssm_tr = data[0][0][0].cuda()
                Xtr_seg_bias = data[1][0][0].cuda()
                wi_seg = np.load("{}/w_{}.npy".format(wi_dir, i), allow_pickle=True)
                wi_seg = torch.cuda.FloatTensor(wi_seg)
                bssm = bssm_sign(Xtr_seg_bias, p_m, is_train=True)
                bssm = (bssm+1)/2 # BSSM -> [0,1]
                
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
                
                e_t = (err * wi_seg).sum()
                
                optimizer.zero_grad()
                e_t.backward()
                optimizer.step()
                ep_losses.append(float(e_t))
                del e_t
                
                if args.debug and i < 3 and (epoch+1) % momentum_range == 0:
                    print("DEBUG. ", epoch, i)
                    print("DEBUG. W: ", get_stats(pt_to_np(wi_seg)))
                    print("DEBUG. Xtr: ", get_stats(pt_to_np(Xtr_seg_bias)))
                    print("DEBUG. SSM: ", get_stats(pt_to_np(ssm_tr)))
                    print("DEBUG. BSSM: ", get_stats(pt_to_np(bssm)))
                    print("DEBUG. Err: ", get_stats(pt_to_np(err)))
                    print("DEBUG. e_t: ", ep_losses[-1])
                
            tr_losses.append(np.mean(ep_losses))
            # Determine whether learner is converging
            if len(tr_losses) > momentum_range:
                prev_momentum = curr_momentum
                curr_momentum = np.mean(tr_losses[-momentum_range:])
                diff = prev_momentum - curr_momentum
                
            if args.debug and (epoch+1) % momentum_range == 0: 
                print("DEBUG. Projection no.{} | epoch {} | mean losses {:.3f} | momentum difference {:.7f}".format(
                    m, epoch, np.mean(tr_losses), diff))
            epoch += 1

        # Update Adaboost parameters at end of training
        beta = get_beta(Xtr_ssm_dl, Xtr_sgbs_dl, wi_dir, p_m, args)
        for i, data in enumerate(zip(Xtr_ssm_dl, Xtr_sgbs_dl)):
            ssm_tr = data[0][0][0].cuda()
            Xtr_seg_bias = data[1][0][0].cuda()
            wi_seg = np.load("{}/w_{}.npy".format(wi_dir, i), allow_pickle=True)
            bssm = bssm_sign(Xtr_seg_bias, p_m, is_train=False)
            bssm = (bssm+1)/2
            
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

            wi_seg = wi_seg*(beta**(1-pt_to_np(err)))
            wi_seg /= wi_seg.sum()+1e-10
            np.save("{}/w_{}".format(wi_dir, i), wi_seg, allow_pickle=True)

        tic = time.time()
        print ("Time: Learning projection #{}: {:.2f} for {} iterations.\n\t beta:{:.3f} diff:{:.7f}".format(
            m+1, tic-toc, epoch, beta, diff))
        if args.debug:
            print ("DEBUG. tr_losses[::20] = ", tr_losses[::20])
        projections.append(p_m.detach().cpu().numpy())
        proj_losses.append(tr_losses)
        betas.append(beta)

        # Saving results
        np.save("{}/projs".format(output_directory), np.array(projections), allow_pickle=True)
        save_pkl(proj_losses, "{}/projlosses.pkl".format(output_directory))
        np.save("{}/betas".format(output_directory), np.array(betas), allow_pickle=True)
        if epoch < 5 or epoch % args.save_every == 0:
            w1s.append(wi_seg)
            np.save("{}/w1s".format(output_directory), np.array(w1s), allow_pickle=True)
    np.save("{}/projs".format(output_directory), np.array(projections), allow_pickle=True)
    save_pkl(proj_losses, "{}/projlosses.pkl".format(output_directory))
    np.save("{}/betas".format(output_directory), np.array(betas), allow_pickle=True)
    np.save("{}/w1s".format(output_directory), np.array(w1s), allow_pickle=True)
    
if __name__ == "__main__":
    main()
