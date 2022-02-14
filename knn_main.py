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
    
    parser.add_argument("--is_save_only", action='store_true',
                        help="Save recon for later evaluation")
    parser.add_argument("--is_create_dir", action='store_true',
                        help="Save recon for later evaluation")
    
    parser.add_argument("--is_jaccard", action='store_true',
                        help="Jaccard Similarity instead of Hamming")
    
    parser.add_argument("--prelim", action='store_true',
                        help="")
    
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
    elif args.use_log_mel:
        Xtr_load_nm = "Xtr_log_Mel.npy"
        Xva_load_nm = "Xva_log_Mel.pkl"
        Xte_load_nm = "Xte_log_Mel.pkl"
    
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
    save_nm = "{}/Fom_{}/results_Prj[{}|{}]_Fom[{}]_See[{}]_Per[{}]_Cld[{}]".format(
        "/media/sdb1/kimsunw/BLSH_Results_Saved", feature_or_model, args.is_proj, args.n_proj, feature_or_model, 
        args.seed, int(args.use_perc*100), args.is_closed)
    if args.is_create_dir:
        os.makedirs(save_nm, exist_ok=True)
    
    print ("Starting script on GPU {}...".format(args.gpu_id))
    print ("Job: {}".format(save_nm))
    
    # Load training dictionary
    Xtr = np.load(Xtr_load_nm)
    Ytr = np.load("IBM_STFT.npy")
    
    # Random sampling
    np.random.seed(args.seed)
    perm_idx = np.random.permutation(len(Xtr))[:int(len(Xtr) * args.use_perc)]
    Xtr = Xtr[perm_idx]
    Ytr = Ytr[perm_idx]
    
    # Load testing data
    if args.is_closed:
        print ("Loading closed set...")
        
        Xva = load_pkl(Xva_load_nm)
        vaX = load_pkl("vaX_STFT.pkl")
        with open('vasnx_wavefiles.pkl', 'rb') as handle:
            val_waves_dict = pickle.load(handle)
            
    else:   
        print ("Loading open set...")
            
        Xva = load_pkl(Xte_load_nm)
        vaX = load_pkl("teX_STFT.pkl")
        with open('tesnx_wavefiles.pkl', 'rb') as handle:
            val_waves_dict = pickle.load(handle)
            
    vas = val_waves_dict['s']
    van = val_waves_dict['n']
    vax = val_waves_dict['x']
    
    # Load projections and apply
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
        
        if args.gpu_id != -1:
            Xtr = torch.cuda.FloatTensor(Xtr)
            Xtr_bias = torch.cat((Xtr, torch.ones((len(Xtr),1)).cuda()), 1)
            projections = torch.cuda.FloatTensor(projections)
            applied_tr = torch.sign(Xtr_bias.mm(projections))
        else:
            Xtr_bias = np.pad(Xtr, [(0, 0), (0, 1)], mode='constant')
            applied_tr = np.sign(np.dot(Xtr_bias, projections))
        
    else:
        # Oracle kNN baseline
        if args.gpu_id != -1:
            applied_tr = torch.cuda.FloatTensor(Xtr)
        else:
            applied_tr = Xtr
        
    # Get scores
    N_va = len(Xva)
    SDRlist=np.zeros(N_va)
    ml=np.zeros(N_va)
    mirSDRlist=np.zeros(N_va)
    mirSIRlist=np.zeros(N_va)
    mirSARlist=np.zeros(N_va)
    
    for i in range(N_va):
        # Apply projections on validation data
        if args.is_proj:
            if args.gpu_id != -1:
                Xva_i = torch.cuda.FloatTensor(Xva[i])
                Xva_i_bias = torch.cat((Xva_i, torch.ones((len(Xva_i),1)).cuda()), 1)
                applied_vate = torch.sign(Xva_i_bias.mm(projections))
            else:
                Xva_i_bias = np.pad(Xva[i], [(0, 0), (0, 1)], mode='constant')
                applied_vate = np.sign(np.dot(Xva_i_bias, projections))
        else:
            if args.gpu_id != -1:
                applied_vate = torch.cuda.FloatTensor(Xva[i])
            else:
                applied_vate = Xva[i]
        
        # Compute similarity scores
        if args.is_jaccard:
            a_ = (applied_tr+1)/2
            b_ = (applied_vate+1)/2
            if args.gpu_id != -1:
                match1s = a_.mm(b_.t())
                match0s = (1-a_).mm(1-b_.t())
            else:
                match1s = a_.dot(b_.T)
                match0s = (1-a_).dot(1-b_.T)
            numers = match1s
            denoms = args.n_proj-match0s
            scores = numers/denoms
            if args.gpu_id != -1:
                scores = scores.detach().cpu().numpy()
        else:
            if args.gpu_id != -1:
                scores = applied_tr.mm(applied_vate.t())
                scores = scores.detach().cpu().numpy()
            else:
                scores = np.dot(applied_tr, applied_vate.T)
            if args.is_proj:
                scores = ((scores+args.n_proj)/2)
        
        
        # Apply average of K IBMs
        K_locs = np.argpartition(-scores, args.K, 0)[:args.K]
        Yhat = np.mean(Ytr[K_locs],0)
        applied = vaX[i] * Yhat
        recon = librosa.istft(applied.T, hop_length=256)
        
        if args.is_save_only:
            np.save("{}/recon_{}".format(save_nm, i), recon)
            if (i+1) % args.print_every == 0:
                prog = (i+1)/len(vas)*100
                print ("{}: {:.2f}% Saved signal.".format(i+1, prog))
            
        else:
            ml[i], SDRlist[i] = SDR(vas[i], recon)
            msdr, msir, msar = get_mir_scores(
                vas[i], van[i], vax[i], recon)
            mirSDRlist[i] = msdr
            mirSIRlist[i] = msir
            mirSARlist[i] = msar

            # Print intermediate results
            if (i+1) % args.print_every == 0:
                curr_tot_SDR = np.sum(ml*SDRlist/np.sum(ml))
                curr_tot_mSDR = np.sum(ml*mirSDRlist/np.sum(ml))
                curr_tot_mSIR = np.sum(ml*mirSIRlist/np.sum(ml))
                curr_tot_mSAR = np.sum(ml*mirSARlist/np.sum(ml))

                prog = (i+1)/len(vas)*100

                print ("{}: {:.2f}% SDR {:.2f} mSDR {:.2f} mSIR {:.2f} mSAR {:.2f}"
                       .format(i+1, prog, curr_tot_SDR, curr_tot_mSDR, 
                               curr_tot_mSIR, curr_tot_mSAR))
                
                if args.prelim and (i+1) == 20:
                    return -1
    
    if not args.is_save_only:
        # Save results
        result_dict = {
            'sdr': SDRlist,
            'msdr': mirSDRlist,
            'msir': mirSIRlist,
            'msar': mirSARlist,
            'ml': ml
        }

        with open('{}/results.pkl'.format(save_nm), 'wb') as handle:
                pickle.dump(result_dict, handle)
        print ("Results saved!")
    
if __name__ == "__main__":
    main()

