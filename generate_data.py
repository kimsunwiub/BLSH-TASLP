from argparse import ArgumentParser
import numpy as np
import librosa
import pickle
import os

from loader import setup_ada_training_data, setup_ada_testing_data
from utils import save_pkl, load_pkl

def parse_arguments():
    parser = ArgumentParser()    
    parser.add_argument("--seed", type=int, default=42,
                        help = "Seed for train and test speaker selection")
    parser.add_argument("--make_wavefiles", action='store_true',
                        help = "Option to generate wavefiles")
    parser.add_argument("--make_stfts", action='store_true',
                        help = "Option to generate STFTs")
    parser.add_argument("--make_mels", action='store_true',
                        help = "Option to generate Mels")
    parser.add_argument("--make_mfccs", action='store_true',
                        help = "Option to generate MFCCs")
    parser.add_argument("--make_log_mels", action='store_true',
                        help = "Option to generate log Mels")
    parser.add_argument("--segment_len", type=int, default=1000,
                        help = "Segment length.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    np.random.seed(args.seed)
    
    if args.make_wavefiles:
        # Create and save wavefiles for specified number of speakers
        trs, trn, trx, vas, van, vax, _ = setup_ada_training_data(args.seed)
        tes, ten, tex, _ = setup_ada_testing_data(args.seed)
        
        # Save files
        train_waves_dict = {'s': trs, 'x': trx, 'n': trn}
        with open('trsnx_wavefiles.pkl', 'wb') as handle:
            pickle.dump(train_waves_dict, handle)
        with open('trs_wavefiles.pkl', 'wb') as handle:
            pickle.dump(trs, handle)
        with open('trx_wavefiles.pkl', 'wb') as handle:
            pickle.dump(trx, handle)
        with open('trn_wavefiles.pkl', 'wb') as handle:
            pickle.dump(trn, handle)
            
        val_waves_dict = {'s': vas, 'x': vax, 'n': van}
        with open('vasnx_wavefiles.pkl', 'wb') as handle:
            pickle.dump(val_waves_dict, handle)
        save_pkl(vax, 'vax_wavefiles.pkl')
        
        test_waves_dict = {'s': tes, 'x': tex, 'n': ten}
        with open('tesnx_wavefiles.pkl', 'wb') as handle:
            pickle.dump(test_waves_dict, handle)
        save_pkl(tex, 'tex_wavefiles.pkl')
    
    if args.make_stfts:
        # Check if wavefiles have been generated
        if not os.path.exists('trsnx_wavefiles.pkl'):
            print ("Need to generate wavefiles first.")
            return -1

        # Load generated wavefiles
        with open('trsnx_wavefiles.pkl', 'rb') as handle:
            train_waves_dict = pickle.load(handle)
        trs = train_waves_dict['s']
        trn = train_waves_dict['n']
        trx = train_waves_dict['x']
        vax = load_pkl('vax_wavefiles.pkl')
        tex = load_pkl('tex_wavefiles.pkl')

        # STFT
        trS = [librosa.stft(x, n_fft=1024, hop_length=256).T for x in trs]
        trN = [librosa.stft(x, n_fft=1024, hop_length=256).T for x in trn]
        trX = [librosa.stft(x, n_fft=1024, hop_length=256).T for x in trx]

        vaX = [librosa.stft(x, n_fft=1024, hop_length=256).T for x in vax]

        teX = [librosa.stft(x, n_fft=1024, hop_length=256).T for x in tex]

        # Magnitude (Float 32 to save complexity)
        trS_mag = [np.abs(stft).astype(np.float32) for stft in trS]
        trN_mag = [np.abs(stft).astype(np.float32) for stft in trN]
        trX_mag = [np.abs(stft).astype(np.float32) for stft in trX]

        vaX_mag = [np.abs(stft).astype(np.float32) for stft in vaX]

        teX_mag = [np.abs(stft).astype(np.float32) for stft in teX]
        
        save_pkl(trX, 'trX_STFT.pkl')
        save_pkl(vaX, 'vaX_STFT.pkl')
        save_pkl(teX, 'teX_STFT.pkl')
        # Only these signals (trX_STFT.pkl) are used for kNN procedure
        # Rest of the datagen is for generating the normed (Xtr_MFCC.npy)
        # for training the weak learners. 
        
        save_pkl(trX_mag, 'trX_mag_STFT.pkl')
        save_pkl(vaX_mag, 'vaX_mag_STFT.pkl')
        save_pkl(teX_mag, 'teX_mag_STFT.pkl')

        # IBM
        IBM = [(trS_mag[i] > trN_mag[i]) for i in range(len(trS_mag))]
        IBM = np.concatenate(IBM, 0)
        np.save("IBM_STFT.npy", IBM)
        
        Xtr = trX_mag
        Xva = vaX_mag
        Xte = teX_mag
        
    if args.make_mels:
        # Check if magnitude STFTs have been generated
        if not os.path.exists('trX_mag_STFT.pkl'):
            print ("Need to generate STFTs first.")
            return -1
        
        trX_mag = load_pkl('trX_mag_STFT.pkl')
        vaX_mag = load_pkl('vaX_mag_STFT.pkl')
        teX_mag = load_pkl('teX_mag_STFT.pkl')
        
        # Mel on power spectrogram    
        Xtr = [librosa.feature.melspectrogram(
                S=x.T**2, sr=16000).T.astype(np.float32) 
                for x in trX_mag]
        Xva = [librosa.feature.melspectrogram(
                S=x.T**2, sr=16000).T.astype(np.float32)  
                for x in vaX_mag]
        Xte = [librosa.feature.melspectrogram(
                S=x.T**2, sr=16000).T.astype(np.float32) 
                for x in teX_mag]
        
        save_pkl(Xtr, 'trX_Mel.pkl')
        save_pkl(Xva, 'vaX_Mel.pkl')
        save_pkl(Xte, 'teX_Mel.pkl')

    if args.make_log_mels:
        # Check if magnitude STFTs have been generated
        if not os.path.exists('trX_mag_STFT.pkl'):
            print ("Need to generate STFTs first.")
            return -1
        
        trX_mag = load_pkl('trX_mag_STFT.pkl')
        vaX_mag = load_pkl('vaX_mag_STFT.pkl')
        teX_mag = load_pkl('teX_mag_STFT.pkl')
        
        # Mel on log power spectrogram    
        # +1 factor to avoid negative values
        Xtr = [librosa.feature.melspectrogram(
                S=np.log(x.T**2+1), sr=16000).T.astype(np.float32) 
                for x in trX_mag]
        Xva = [librosa.feature.melspectrogram(
                S=np.log(x.T**2+1), sr=16000).T.astype(np.float32)  
                for x in vaX_mag]
        Xte = [librosa.feature.melspectrogram(
                S=np.log(x.T**2+1), sr=16000).T.astype(np.float32) 
                for x in teX_mag]
        
        save_pkl(Xtr, 'trX_log_Mel.pkl')
        save_pkl(Xva, 'vaX_log_Mel.pkl')
        save_pkl(Xte, 'teX_log_Mel.pkl')
        
    if args.make_mfccs:
        # Check if Mels have been generated
        if not os.path.exists('trX_Mel.pkl'):
            print ("Need to generate Mels first.")
            return -1
        
        trX_Mel = load_pkl('trX_Mel.pkl')
        vaX_Mel = load_pkl('vaX_Mel.pkl')
        teX_Mel = load_pkl('teX_Mel.pkl')
        
        # Mel-frequency cepstral coefficients
        Xtr = [librosa.feature.mfcc(
                S=librosa.power_to_db(x).T, sr=16000).T.astype(np.float32) 
                for x in trX_Mel]
        Xva = [librosa.feature.mfcc(
                S=librosa.power_to_db(x).T, sr=16000).T.astype(np.float32)  
                for x in vaX_Mel]
        Xte = [librosa.feature.mfcc(
                S=librosa.power_to_db(x).T, sr=16000).T.astype(np.float32)
                for x in teX_Mel]
    
    if args.make_stfts or args.make_mels or args.make_mfccs or args.make_log_mels:
        Xtr = np.concatenate(Xtr, 0)

        # Normalize
        Xtr = Xtr/np.linalg.norm(Xtr+1e-4, axis=1)[:,None]
        Xva = [x/np.linalg.norm(x+1e-4, axis=1)[:,None] for x in Xva]
        Xte = [x/np.linalg.norm(x+1e-4, axis=1)[:,None] for x in Xte]

        # Save files
        if args.make_mels: 
            suffix = "Mel"
        elif args.make_mfccs: 
            suffix = "MFCC"
        elif args.make_log_mels: 
            suffix = "log_Mel"
        else: 
            suffix = "STFT"
        np.save("Xtr_{}".format(suffix), Xtr)
        save_pkl(Xva, "Xva_{}.pkl".format(suffix))
        save_pkl(Xte, "Xte_{}.pkl".format(suffix))
        
        # Concatenated
        Xva_concat = np.concatenate(Xva, 0)
        np.save("Xva_{}".format(suffix), Xva_concat)
        
if __name__ == "__main__":
    main()
