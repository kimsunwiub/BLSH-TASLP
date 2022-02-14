from argparse import ArgumentParser
import numpy as np
import librosa
import pickle

import torch
torch.set_num_threads(1)

import os

from utils import SDR, get_mir_scores, load_pkl

def parse_arguments():
    parser = ArgumentParser() 
    
    parser.add_argument("-p", "--is_proj", action='store_true',
                        help="kNN on projections")
    parser.add_argument("--use_stft", action='store_true',
                        help="Using STFT features")
    parser.add_argument("--use_mel", action='store_true',
                        help="Using MFCC features")
    parser.add_argument("--use_log_mel", action='store_true',
                        help="Using log Mel features")
    parser.add_argument("--is_closed", action='store_true',
                        help="Open (Test) / Closed (Val)")
    
    parser.add_argument("--load_model", type=str, default=None,
                        help="Trained projections")    
    parser.add_argument("-n", "--n_proj", type=int, default=None,
                        help = "Number of projections")
    parser.add_argument("--use_perc", type=float, default=.1,
                        help = "Random sample %% of training set")
    parser.add_argument("-k", "--K", type=int, default=10,
                        help = "Number of neighbors")
    parser.add_argument("--seed", type=int, default=42,
                        help = "Seed for random sampling from dictionary")
    parser.add_argument("--gpu_id", type=int, default=-1,
                        help = "GPU ID. -1 for ")
    parser.add_argument("--print_every", type=int, default=100,
                        help = "Option for printing frequency")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if args.gpu_id != -1:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Decide features to work on (STFT, Mel (M=128), or MFCC)
    if args.use_stft:
        Xtr_load_nm = "Xtr_STFT.npy"
        Xva_load_nm = "Xva_STFT.pkl"
        Xte_load_nm = "Xte_STFT.pkl"
    elif args.use_mel:
        Xtr_load_nm = "Xtr_Mel.npy"
        Xva_load_nm = "Xva_Mel.pkl"
        Xte_load_nm = "Xte_Mel.pkl"
    
    # Name for saving results
    feature_or_model = Xtr_load_nm.split('.')[0]
    if args.load_model: 
        feature_or_model = 'expr' + args.load_model.split('expr')[1]
        feature_or_model = feature_or_model.replace('/', '')
    else:
        if args.is_proj:
            feature_or_model += '_LSH'
        else:
            feature_or_model += '_kNN'
    save_nm = "scores/Fom_{}/results_Prj[{}|{}]_Fom[{}]_See[{}]_Per[{}]_Cld[{}]".format(
        feature_or_model, args.is_proj, args.n_proj, feature_or_model, 
        args.seed, int(args.use_perc*100), args.is_closed)
    
    os.makedirs(save_nm, exist_ok=True)
    f = open("{}/scores.txt".format(save_nm), "a")
    print ("Starting script on GPU {}...".format(args.gpu_id))
    print_line = "Job: {}\n".format(save_nm)
    print (print_line)
    f.write(print_line)
    
    # Load training dictionary
    Xtr = np.load(Xtr_load_nm)
    Xva = load_pkl(Xte_load_nm)
    
    # Random sampling
    np.random.seed(args.seed)
    perm_idx = np.random.permutation(len(Xtr))[:int(len(Xtr) * args.use_perc)]
    Xtr = Xtr[perm_idx]
    Xtr = torch.cuda.FloatTensor(Xtr)
    if args.is_proj:
        print ("Loading projections, ", args.load_model)
        if args.load_model:
            projections = np.load(
                "{}/projs.npy".format(args.load_model))
            projections = projections.squeeze().T
            print ("Loaded Shape: ", projections.shape)
            projections = projections[:,:args.n_proj]
        else:
            # LSH Random projection baseline
            np.random.seed(args.seed)
            # projections = np.random.rand(Xtr.shape[1], args.n_proj)
            projections = np.random.normal(loc=0.0, 
                            scale=1./args.n_proj, 
                            size=(Xtr.shape[1]+1, args.n_proj))

        print ("Projections Shape: ", projections.shape)
        Xtr_bias = torch.cat((Xtr, torch.ones((len(Xtr),1)).cuda()), 1)
        projections = torch.cuda.FloatTensor(projections)
        applied_tr = torch.sign(Xtr_bias.mm(projections))
        
    # Get scores
    N_va = len(Xva)
    hamm_list = np.zeros(N_va)
    cos_list = np.zeros(N_va)
    hamm = 0.0
    cos = 0.0
    for i in range(N_va):
        Xva_i = torch.cuda.FloatTensor(Xva[i])
        scores_float = Xtr.mm(Xva_i.t())
        scores_float = scores_float.detach().cpu().numpy()
        if args.is_proj:
            Xva_i_bias = torch.cat((Xva_i, torch.ones((len(Xva_i),1)).cuda()), 1)
            applied_vate = torch.sign(Xva_i_bias.mm(projections))
            scores = applied_tr.mm(applied_vate.t())
            scores = scores.detach().cpu().numpy()
            scores = ((scores+args.n_proj)/2)
            K_locs = np.argpartition(-scores, args.K, 0)[:args.K]
            hamm = scores[K_locs[:,0],0].mean()
            hamm_list[i] = hamm
        else:
            K_locs = np.argpartition(-scores_float, args.K, 0)[:args.K]
        cos = scores_float[K_locs[:,0],0].mean()        
        cos_list[i] = cos
        
        if (i+1) % args.print_every == 0:
            prog = (i+1)/len(Xva)*100
            curr_tot_hamm = np.sum(hamm_list)/(i+1)
            curr_tot_cos = np.sum(cos_list)/(i+1)
            print_line = "{}: {:.2f}% Hamm {:.2f} Cos {:.3f}".format(i+1, prog, curr_tot_hamm, curr_tot_cos)
            print (print_line)
            print_line += "\n"
            f.write(print_line)
    
    f.write("END\n")
    f.close()
    
if __name__ == "__main__":
    main()

