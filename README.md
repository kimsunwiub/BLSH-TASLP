# Boosted Locality Sensitive Hashing: Discriminative, Efficient, and Scalable Binary Codes for Source Separation
This repository provides scripts required for training "Boosted Locality Sensitive Hash (BLSH)" functions and the kNN-based source separation algorithm. 

## Description

#### loader.py
* Helper functions to load TIMIT speakers and Duan and DEMAND noise datasets.

#### generate_data.py
* Generate training and testing wavefiles and spectral features.

#### generate_dataloaders.py
* Partition generated data into separate folders to call using a dataloader.

#### train_weak_learners.py
* A python script to train weak learners on various features. 

#### knn_main.py
* Perform KNN using trained weak learners and save the reconstructions of estimated clean signals. 
* Can also compute BSS Eval scores (SDR, SIR, and SAR).

#### get_perf_afterknn.py
* Compute SDR, SISNR, PESQ, and ESTOI metrics from saved reconstructions. 

#### utils.py
* Helper functions for training and evaluation.

#### dnn_utils.py
* Helper functions for training and evaluating DNN baselines.

#### dnn_models.py
* Fully connected and RNN model definitions.

#### dnn_data.py
* Helper functions to load datasets.

#### dnn_train.py
* To train DNN models.

#### dnn_test.py
* To evaluate DNN models.


## Usage
Make sure that ```pip``` and ```python3``` are installed (The program was written using Python 3.6) and install the script's dependencies. Note: ```Librosa``` is used for audio reading and writing but can be replaced with other packages such as ```scipy.signal```. ```Matplotlib``` can be removed if not plotting the results. 

### Data Generation
* To generate the clean source, noises, and mixture wavefiles and the spectral features. e.g.,
```
python generate_data.py --make_wavefiles --make_stfts
```
to generate and save the wavefiles and STFT spectrograms. 

* Then further separate the generated data into folders to load using PyTorch Dataloaders. e.g.,
```
python generate_dataloaders.py --use_stft --use_rbf_target --sigma2 0.9 --gpu 3
```
to generate the SSMs using RBF kernels with width 0.9. 

### Training weak learners
* To construct hash functions in the form of weak learners. e.g.,
```
python train_weak_learners.py --use_stft_learner --use_stft_target --use_rbf_target --sigma2 0.9 --proj_lossfn xent --proj_target_scale_zero --ada_target_scale_zero --gpu_id 1 --save_model Saved_Models
```
to train weak learners using STFT feature frames and RBF (width 0.9) SSM targets. Cross entropy loss function is used. All bipolar binary values are scaled between [0,1]. 

### Evaluation

#### kNN Procedure
* To test the performance of kNN procedure using ground truth STFT dictionary on closed set. e.g.,
```
python knn_main.py --use_stft -p --use_perc 0.1 -n 50 --load_model Saved_Models/expr03010904_feat(STFT_STFT)_kern(rbf_0.9)_lr(1e-04)_loss(xent_sse)_scale(True_True)_GPU1/ --gpu_id 4 --is_save_only --is_create_dir
```
to perform kNN procedure using 0.1 partition of the STFT feature database and first 50 weak learners trained on RBF (width 0.9) BSSM targets. 
* The --is_save_only option saves the reconstructions of clean estimates for further evaluation. 

#### Further evaluation
* For further evaluation, e.g.,
```
python get_perf_afterknn.py --sdr --sisnr --pesq --estoi --results_dir BLSH_Results_Saved/Fom_expr02162151_feat(STFT_STFT)_kern(rbf_0.9)_lr(1e-04)_loss(xent_sse)_scale(True)_GPU4/ -n 50 -p 0.1 
```
to get SDR, SISNR, PESQ, and ESTOI scores using reconstructions from the previous step. 

### DNN Baselines
* To train DNN baseline models as reported in the manuscript. e.g.,
```
python dnn_train.py -l 3 -u 32 --device 4 --fc --learning_rate 1e-4 --b1 0.9 --b2 0.999 --is_save
```
to train a 3x32 fully connected network. 

* To evaluate DNN models. e.g.,
```
python dnn_test.py -l 3 -u 32 --device 5 --fc
```
to load and evaluate a trained 3x32 fully connected model.

### Datasets used in this repository
* TIMIT (https://catalog.ldc.upenn.edu/LDC93S1)
* Duan (http://www2.ece.rochester.edu/~zduan/data/noise/)
* DEMAND (https://zenodo.org/record/1227121#.Xbm4X797leg)

## Acknowledgements
This material is based upon work supported by the National Science Foundation under Award Number:1909509.
