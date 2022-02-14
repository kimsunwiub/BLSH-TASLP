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
    parser.add_argument("--use_logMel_learner", action='store_true', 
        help="Using STFT features. Otherwise, log mel features are used.")
    parser.add_argument("--use_logMel_target", action='store_true', 
        help="Using STFT target. Otherwise, log mel targets are used.")
    parser.add_argument("--use_mel_learner", action='store_true', 
        help="Using STFT features. Otherwise, log mel features are used.")
    parser.add_argument("--use_mel_target", action='store_true', 
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
    parser.add_argument("--sigma2", type=float, default=0.0, 
        help="Denominator value for RBF kernel")
    parser.add_argument("--save_every", type=int, default=10,
        help = "Specify saving frequency")
    
    parser.add_argument("--proj_target_scale_zero", action='store_true',
        help="True: Target value ranges [0,1]. False: [-1,1]")
    parser.add_argument("--ada_target_scale_zero", action='store_true',
        help="True: Target value ranges [0,1]. False: [-1,1]")
    parser.add_argument("--proj_lossfn", type=str, default='sse',
        help="Loss function for training weak learners. Default: 'sse'. Other: 'xent' if proj_target_scale_zero is True")
    parser.add_argument("--ada_lossfn", type=str, default='sse',
        help="Loss function for updating observation weights. Default: 'sse'. Other: 'mult' if ada_target_scale_zero is False")
    
    parser.add_argument("--for_plot", action='store_true',
        help="True: Save info for a segment")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
    args.kernel = 'rbf' if args.use_rbf_target else 'linear' # SSM
        
    """
    Datasets and Dataloaders:
    SSM and frames are loaded by a 1000x1000 SSM size and 1000xF frames. For 
    weak learners to be discriminative, the SSM and frames have been shuffled, 
    which the dataloaders then load in order.
    """
    Xtr_ssm_dl = load_ssm_dataloader(args)
    Xtr_sgbs_dl = load_sgbs_dataloader(args)
 
    n_features = 513 if args.use_stft_learner else 128
    Ntr = len(Xtr_ssm_dl)
    print(Ntr)
    
    if args.use_stft_learner:
        feat = "STFT"
    elif args.use_logMel_learner:
        feat = "logMel"
    elif args.use_mel_learner:
        feat = "Mel"
    feat += "_"
    if args.use_stft_target:
        feat += "STFT"
    elif args.use_logMel_target:
        feat += "logMel"
    elif args.use_mel_target:
        feat += "Mel"
    
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
        model_nm = "feat({})_kern({}_{})_lr({:.0e})_loss({}_{})_scale({}_{})".format(
            feat, args.kernel, args.sigma2, args.lr, args.proj_lossfn, 
            args.ada_lossfn, args.proj_target_scale_zero, args.ada_target_scale_zero)
        output_directory = "{}/expr{}_{}_GPU{}".format(
            args.save_model, t_stamp, model_nm, args.gpu_id)
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
        
    if args.for_plot:
        if args.use_mel_target:
            plot_Xtr_sgbs = np.load("Xtr_Mel_plot_seg.npy")
        else:
            plot_Xtr_sgbs = np.load("Xtr_STFT_plot_seg.npy")
        plot_Xtr_ssm = plot_Xtr_sgbs.dot(plot_Xtr_sgbs.T)
        plot_sgbs_bias = np.concatenate((plot_Xtr_sgbs, np.ones((len(plot_Xtr_sgbs),1))),1).astype(np.float32)
        plot_wi = np.ones((300,300))
        init_val = 1.0/(300**2)
        plot_wi *= init_val
    
    """
    Convergence for training projections is determined by the saturation or increase
    of the moving average of training losses over 10 epochs. Validation set is not
    used due to observation weights corresponding only to training examples. Training
    is run for a minimum of 20 epochs and a maximum of 200 epochs. 
    """
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
        while epoch < args.min_iter or (diff > tol and epoch < args.max_iter):
            ep_losses = []
            # Training
            for i, data in enumerate(zip(Xtr_ssm_dl, Xtr_sgbs_dl)):
                ssm = data[0][0][0].float().cuda()
                Xtr_seg_bias = data[1][0][0].float().cuda()
                wi_seg = np.load("{}/w_{}.npy".format(wi_dir, i), allow_pickle=True)
                wi_seg = torch.cuda.FloatTensor(wi_seg)
                bssm = bssm_sign(Xtr_seg_bias, p_m, is_train=True)
                
                if args.proj_target_scale_zero:
                    bssm = (bssm+1)/2
                else:
                    ssm = ssm*2-1                    
                    
                if args.proj_lossfn == 'xent':
                    assert args.proj_target_scale_zero
                    err = xent_fn(bssm,ssm)
                elif args.proj_lossfn == 'sse':
                    err = (bssm-ssm)**2
                
                e_t = (err * wi_seg).sum()
                optimizer.zero_grad()
                e_t.backward()
                optimizer.step()
                ep_losses.append(float(e_t))
                del e_t
                
            tr_losses.append(np.mean(ep_losses))
            # Determine whether learner is converging
            if len(tr_losses) > momentum_range:
                prev_momentum = curr_momentum
                curr_momentum = np.mean(tr_losses[-momentum_range:])
                diff = prev_momentum - curr_momentum
                
            if (epoch+1) % momentum_range == 0: 
                print("DEBUG. Projection no.{} | epoch {} | mean losses {:.3f} | momentum difference {:.7f}".format(
                    m, epoch, np.mean(tr_losses), diff))
            epoch += 1

        # Update Adaboost parameters at end of training
        beta = get_beta(Xtr_ssm_dl, Xtr_sgbs_dl, wi_dir, p_m, args)
        for i, data in enumerate(zip(Xtr_ssm_dl, Xtr_sgbs_dl)):
            ssm = data[0][0][0].float().cuda()
            Xtr_seg_bias = data[1][0][0].float().cuda()
            wi_seg = np.load("{}/w_{}.npy".format(wi_dir, i), allow_pickle=True)
            bssm = bssm_sign(Xtr_seg_bias, p_m, is_train=False)
            
            if args.ada_target_scale_zero:
                assert args.ada_target_scale_zero
                bssm = (bssm+1)/2
            else:
                ssm = ssm*2-1

            """
            Loss functions: 
            
            Scale: [0,1]
            Proj = SSE or Xent
            Ada = -SSE or -Xent (too sharp)

            Scale: [-1,1]
            Proj = SSE
            Ada = -SSE (too sharp) or y*yhat (better. == l1 without /2).
            """
                
            if args.ada_lossfn == 'xent':
                assert args.ada_target_scale_zero
                err = -xent_fn(bssm,ssm)
            elif args.ada_lossfn == 'mult':
                assert not args.ada_target_scale_zero
                err = ssm*bssm
            elif args.ada_lossfn == 'sse':
                err = -(bssm-ssm)**2
                
            if i < 3:
                print("DEBUG. Proj no.{}, Iter {}".format(m, i))
                print("DEBUG. W>1e-5: ", np.sum(wi_seg>1e-5))
                print("DEBUG. W: ", wi_seg[500,500:505])
                print("DEBUG. SSM: ", pt_to_np(ssm)[500,500:505])
                print("DEBUG. BSSM: ", pt_to_np(bssm)[500,500:505])
                print("DEBUG. Err: ", pt_to_np(err)[500,500:505])
                print("DEBUG. Beta: ", beta)
                
            wi_seg = wi_seg*np.exp(-beta*pt_to_np(err))
            wi_seg /= wi_seg.sum()+1e-10
            
            if i < 3:
                print("DEBUG. uW: ", wi_seg[500,500:505])
            np.save("{}/w_{}".format(wi_dir, i), wi_seg, allow_pickle=True)

        tic = time.time()
        print ("Time: Learning projection #{}: {:.2f} for {} iterations.\n\t beta:{:.3f} diff:{:.7f}".format(
            m+1, tic-toc, epoch, beta, diff))
        print ("DEBUG. tr_losses[::20] = ", tr_losses[::20])
        projections.append(p_m.detach().cpu().numpy())
        proj_losses.append(tr_losses)
        betas.append(beta)

        if args.for_plot:
            Xtr_seg_bias = torch.cuda.FloatTensor(plot_sgbs_bias)
            bssm = bssm_sign(Xtr_seg_bias, p_m, is_train=False)
            bssm = (bssm+1)/2
            ssm = torch.cuda.FloatTensor(plot_Xtr_ssm)
            err = -(bssm-ssm)**2
            wi_seg = plot_wi
            wi_seg = wi_seg*np.exp(-beta*pt_to_np(err))
            wi_seg /= wi_seg.sum()+1e-10
            np.save("{}/weight_{}".format(output_directory, m), wi_seg)
            np.save("{}/bssm_{}".format(output_directory, m), pt_to_np(bssm))
            np.save("{}/err_{}".format(output_directory, m), pt_to_np(err))
        
        
        # Saving results
        if (m+1) % args.save_every == 0 or m < 5:
            np.save("{}/projs".format(output_directory), np.array(projections), allow_pickle=True)
            save_pkl(proj_losses, "{}/projlosses.pkl".format(output_directory))
            np.save("{}/betas".format(output_directory), np.array(betas), allow_pickle=True)
            w1s.append(wi_seg)
            np.save("{}/w1s".format(output_directory), np.array(w1s), allow_pickle=True)
    np.save("{}/projs".format(output_directory), np.array(projections), allow_pickle=True)
    save_pkl(proj_losses, "{}/projlosses.pkl".format(output_directory))
    np.save("{}/betas".format(output_directory), np.array(betas), allow_pickle=True)
    np.save("{}/w1s".format(output_directory), np.array(w1s), allow_pickle=True)
    
if __name__ == "__main__":
    main()
