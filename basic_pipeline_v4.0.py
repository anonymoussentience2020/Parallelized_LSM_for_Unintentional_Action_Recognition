import torch
import torch.optim as optim
import torch.nn as nn

import time
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import csv
import sys
import warnings
warnings.filterwarnings('ignore')

from dataloader import get_video_loader
from CNNs import get_CNN
from Spike_Encoding.spike_encoding import spike_encoding 
from LSMs.PLSM_v6 import PLSM as LSM

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#STDP pipeline#---------------------------------------------------------------------------------------------------------

def get_piecewise_mean_spike_count_data(spike_train, window_size = 32):
	assert spike_train.size(1)%window_size == 0

	output = torch.zeros(spike_train.size(0), int(spike_train.size(1)/window_size))
	for i in range(output.size(1)):
		output[:,i] = torch.mean(spike_train[:, window_size*i:window_size*(i+1)], dim=1)
	return output

def get_MRI_data(spike_train):
	MRI_data = []
	for t in range(spike_train.size(1)):
		MRI_data.append(spike_train[:,t].view(LSM.depth, LSM.height, LSM.width).unsqueeze(0))
	return torch.cat(MRI_data, dim=0)

def generate_MRI_dataset(dataloader, step_between_clips, frames_per_clip, Te, start_id=0, val=False):
	#File to save dataset to
	if not val:
		autoencoded_path = 'autoencoded_data/train/'
		cubic_path_1 = 'MRI_data/PLSM_readout_100timesteps/train/'
	else:
		autoencoded_path = 'autoencoded_data/val/'
		cubic_path_1 = 'MRI_data/PLSM_readout_100timesteps/val/'

	video_idx, total_clip_idx = start_id, start_id

	for x,y in tqdm(dataloader, total=len(dataloader)):
	
		filename = str(video_idx)

		clips = x[0].to(device)
		print('Video size:', clips.size())

		LSM.reset()
		ST_buffer = 0

		autoencoded_x = []
		video_1, video_2, video_3 = [], [], []
		for clip_idx,clip in enumerate(clips):
			clip = clip[-step_between_clips:] if clip_idx!=0 else clip
			
			#auto-encode
			auto_encoded_clip = cnn(clip).cpu().numpy()
			autoencoded_x.append(auto_encoded_clip)

			#spike_encode
			clip_spike_train = spike_train_encoder.encode(auto_encoded_clip.T)
			del auto_encoded_clip
			
			#feed to LSM
			activation = LSM.predict(clip_spike_train, output='ST_activation')	#Returns (N X ((fpc or sbc) x T_enc))
			scan = torch.cat([ST_buffer, activation.cpu()], dim=1) if clip_idx!=0 else activation.cpu()
			ST_buffer = scan[:,-(frames_per_clip-step_between_clips)*Te:]	
			del activation

			#get piece-wise mean spike_count
			downsampled_scan_1 = get_piecewise_mean_spike_count_data(scan, window_size = 8)
			del scan
			MRI_1 = get_MRI_data(downsampled_scan_1)
			del downsampled_scan_1
			#save MRI data
			video_1.append(MRI_1.unsqueeze(0).numpy())
			del MRI_1

			total_clip_idx += 1

		y = np.asarray(y)
		data = np.empty(2, dtype=object)

		video_1  = np.concatenate(video_1)
		data[:] = [video_1, y]
		np.save(cubic_path_1+filename+'.npy', data)
		
		video_idx += 1	

if __name__ == '__main__':

	device = torch.device('cuda:0')

	print('\nUsing : ',device)
	print('Ensure that you have created two empty folders named 'train' and 'val' inside location MRI_data/PLSM_readout_100timesteps/')
	
	print('Path example : home/username/Downlaods/Parallelized_LSM_for_Unintentional_Action_Recognition')
	base_path = input('Enter absolute path to the PLSM folder (including PLSM folder name) as mentioned in the example above:')
	      
	annotations_path = os.path.join(base_path,'PATH/TO/annotations/transition_times.json')
	datapath = os.path.join(base_path,'PATH/TO/datasets')
	cnn_checkpoint_path = 'saved_models/auto_encoder_512_completeData_stateDict.pt'

	#Dataloaders : 16fps, fpc = 16 frames, sbc = 4 frames
	fps = 16
	frames_per_clip = 16
	step_between_clips = 4
	train_start_id, val_start_id = 0, 0
	train_loader = get_video_loader(datapath=datapath, annotations_path=annotations_path, val=False, get_video_wise=True, num_workers=0, fps=fps, frames_per_clip=frames_per_clip, step_between_clips=step_between_clips/fps, start_id=train_start_id)
	val_loader = get_video_loader(datapath=datapath, annotations_path=annotations_path, val=True, get_video_wise=True, num_workers=0, fps=fps, frames_per_clip=frames_per_clip, step_between_clips=step_between_clips/fps, start_id=val_start_id)			

	#Load trained autoencoder weights and freeze autoencoder
	cnn = get_CNN()
	cnn = cnn.to(device)
	cnn.load_model(cnn_checkpoint_path)
	cnn.freeze_autoencoder()
	cnn.eval()

	#4: Initialize spike-encoder and LSMs
	T_enc = 50
	input_range = (0.0, 50)#(min_val, max_val)#(0.0, 1.0)#(0.0, 46.3)#
	output_freq_range = (20, 200)	#(50ms, 5ms)
	spike_train_encoder = spike_encoding(scheme='poisson_rate_coding', time_window=T_enc, input_range=input_range, output_freq_range=output_freq_range)

	lsm_name = 'ablation_LSM_10.pt'
	if os.path.exists(os.path.join('saved_models',lsm_name)):
		LSM = torch.load(os.path.join('saved_models',lsm_name))
		print('\nLoaded {} from cache.'.format(lsm_name))
	else:
		print('\nInitializing LSM...')
		LSM = LSM(input_size=512, width=10, height=10, depth=10, output_size=1, 
					neuron_type_ratio = 0.8, input_feature_selection_density=0.1, primary_to_auxiliary_ratio=0.5, 
					W_scale=0.01, lamda=6, version='torch', 
					LIF_Vth=0.1, LIF_tau_m=5, device=device, delay_device=device)
		torch.save(LSM, os.path.join('saved_models',lsm_name))
	print('\nSpike Encoder and LSM initialized.')

	#Generate dataset : feed-forward through LSM-pipeline
	generate_MRI_dataset(dataloader=train_loader, step_between_clips=step_between_clips, frames_per_clip=frames_per_clip, Te=T_enc, val=False, start_id=train_start_id)
	generate_MRI_dataset(dataloader=val_loader, step_between_clips=step_between_clips, frames_per_clip=frames_per_clip, Te=T_enc, val=True, start_id=val_start_id)

		
