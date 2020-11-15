import json
import os
import random
import statistics
from argparse import Namespace
from glob import glob

import sys
sys.path.append('/home/cvpr/Documents/Dipayan/oops-master') #To include libs from oops/master

import av
import torch
import torch.utils.data as data
import torchvision

from torch.utils.data import ConcatDataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm

import py12transforms as T


normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989])
unnormalize = T.Unnormalize(mean=[0.43216, 0.394666, 0.37645],
                            std=[0.22803, 0.22145, 0.216989])
train_transform = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((224,224)),
    T.RandomHorizontalFlip(),
    normalize,
    #T.RandomCrop((224, 224))
])
test_transform = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((224,224)),
    normalize,
    #T.CenterCrop((224, 224))
])

class Dataset(VisionDataset):
    def __init__(self, datapath, annotations_path, transforms, 
        cached_all_train_data_name='cached_all_train_data.pt', cached_valid_train_data_name='cached_valid_train_data.pt', 
        cached_all_val_data_name='cached_all_val_data.pt', cached_valid_val_data_name='cached_valid_val_data.pt',
        get_video_wise=False, val=False, fps=None, frames_per_clip=None, step_between_clips=None, start_id=0):

        self.get_video_wise = get_video_wise
        self.start_id = start_id
        self.transforms = transforms

        #Load annotations = fails_data(in original file)
        with open(annotations_path) as f:
            self.annotations = json.load(f)

        #Load videos
        if fps is None:
            fps = 16
        if frames_per_clip is None:
            frames_per_clip = fps
        if step_between_clips is None:  
            step_between_clips = int(fps * 0.25)    # FPS X seconds = frames
        else:          
            step_between_clips = int(fps * step_between_clips)    # FPS X seconds = frames
        
        #For train_data
        if not val:

            if os.path.exists(os.path.join(datapath,'train',cached_valid_train_data_name)):
                self.video_clips = torch.load(os.path.join(datapath,'train',cached_valid_train_data_name)) 
                print('\nLoaded Valid train data from cache...')
            else:
                #Load all train data
                all_video_list = glob(os.path.join(datapath, 'train', '**', '*.mp4'), recursive=True)

                if os.path.exists(os.path.join(datapath,'train',cached_all_train_data_name)):
                    self.all_video_clips = torch.load(os.path.join(datapath,'train',cached_all_train_data_name))
                    print('\nLoaded all train data from cache...')
                else:
                    print('\nProcessing all train data...')
                    self.all_video_clips = VideoClips(all_video_list, frames_per_clip, step_between_clips, fps)
                    torch.save(self.all_video_clips, os.path.join(datapath,'train',cached_all_train_data_name))

                #Separate out all valid videos  
                print('\nSEPARATING VALID VIDEOS... VAL=',val)
                valid_video_paths = []
                print('Computing all clips...')
                self.all_video_clips.compute_clips(frames_per_clip, step_between_clips, fps)
                for video_idx, vid_clips in tqdm(enumerate(self.all_video_clips.clips), total=len(self.all_video_clips.clips)):
                    video_path = self.all_video_clips.video_paths[video_idx]
                    
                    #Ignore if annotation doesnt exist
                    if os.path.splitext(os.path.basename(video_path))[0] not in self.annotations:
                        continue
                    #Ignore if moov atom error
                    try:
                        #Ignore if video attribute doesnt qualify
                        t_unit = av.open(video_path, metadata_errors='ignore').streams[0].time_base
                        t_fail = sorted(self.annotations[os.path.splitext(os.path.basename(video_path))[0]]['t'])
                        t_fail = t_fail[len(t_fail) // 2]
                        if t_fail < 0 or not 0.01 <= statistics.median(self.annotations[os.path.splitext(os.path.basename(video_path))[0]]['rel_t']) <= 0.99 or \
                                                    self.annotations[os.path.splitext(os.path.basename(video_path))[0]]['len'] < 3.2 or \
                                                    self.annotations[os.path.splitext(os.path.basename(video_path))[0]]['len'] > 30:
                            continue
                    except:
                        continue            
                    #If none of the above happens, then save the video path
                    valid_video_paths.append(video_path)

                self.video_clips = VideoClips(valid_video_paths, frames_per_clip, step_between_clips, fps)
                torch.save(self.video_clips, os.path.join(datapath,'train',cached_valid_train_data_name))
                print('Saved valid train data in cache.')

        #For test data
        else:        
            if os.path.exists(os.path.join(datapath,'val',cached_valid_val_data_name)):
                self.video_clips = torch.load(os.path.join(datapath,'val',cached_valid_val_data_name)) 
                print('\nLoaded Valid Val data from cache...')
            else:
                #Load all val data
                all_video_list = glob(os.path.join(datapath, 'val', '**', '*.mp4'), recursive=True)

                if os.path.exists(os.path.join(datapath,'val',cached_all_val_data_name)):
                    self.all_video_clips = torch.load(os.path.join(datapath,'val',cached_all_val_data_name))
                    print('\nLoaded all val data from cache...')
                else:
                    print('\nProcessing all val data...')
                    self.all_video_clips = VideoClips(all_video_list, frames_per_clip, step_between_clips, fps)
                    torch.save(self.all_video_clips, os.path.join(datapath,'val',cached_all_val_data_name))

                #Separate out all valid videos  
                print('\nSEPARATING VALID VIDEOS... VAL=',val)
                valid_video_paths = []
                print('Computing all clips...')
                self.all_video_clips.compute_clips(frames_per_clip, step_between_clips, fps)
                for video_idx, vid_clips in tqdm(enumerate(self.all_video_clips.clips), total=len(self.all_video_clips.clips)):
                    video_path = self.all_video_clips.video_paths[video_idx]
                    
                    #Ignore if annotation doesnt exist
                    if os.path.splitext(os.path.basename(video_path))[0] not in self.annotations:
                        continue
                    
                    #Ignore if moov atom error
                    try:
                        #Ignore if video attribute doesnt qualify
                        t_unit = av.open(video_path, metadata_errors='ignore').streams[0].time_base
                        t_fail = sorted(self.annotations[os.path.splitext(os.path.basename(video_path))[0]]['t'])
                        t_fail = t_fail[len(t_fail) // 2]
                        if t_fail < 0 or not 0.01 <= statistics.median(self.annotations[os.path.splitext(os.path.basename(video_path))[0]]['rel_t']) <= 0.99 or \
                                                    self.annotations[os.path.splitext(os.path.basename(video_path))[0]]['len'] < 3.2 or \
                                                    self.annotations[os.path.splitext(os.path.basename(video_path))[0]]['len'] > 30:
                            continue
                    except:
                        continue

                    #if moov atom exception occurs then ignore clip
                    try:
                        temp = av.open(video_path, metadata_errors='ignore').streams[0].time_base
                    except:
                        continue

                    #Ignore video attributes for test data : Like video_len and median(rel_t)  
                             
                    valid_video_paths.append(video_path)

                self.video_clips = VideoClips(valid_video_paths, frames_per_clip, step_between_clips, fps)
                torch.save(self.video_clips, os.path.join(datapath,'val',cached_valid_val_data_name))
                print('Saved valid val data in cache.')

        #Load borders.json : LATER

        #Generate all mini-clips of size frames_per_clip from all video clips
        print('\nGenerating VALID mini-clips and labels from',len(self.video_clips.clips),'videos... VAL=',val)
        self.video_clips.compute_clips(frames_per_clip, step_between_clips, fps)
        self.video_clips.labels = []
        for video_idx, vid_clips in tqdm(enumerate(self.video_clips.clips), total=len(self.video_clips.clips)):

            video_path = self.video_clips.video_paths[video_idx]
           
            t_unit = av.open(video_path, metadata_errors='ignore').streams[0].time_base
            t_fail = sorted(self.annotations[os.path.splitext(os.path.basename(video_path))[0]]['t'])
            t_fail = t_fail[len(t_fail) // 2]                
            prev_label = 0
            first_one_idx = len(vid_clips)
            first_two_idx = len(vid_clips)
            for clip_idx, clip in enumerate(vid_clips): #clip == timestamps
                start_pts = clip[0].item()
                end_pts = clip[-1].item()
                t_start = float(t_unit * start_pts)
                t_end = float(t_unit * end_pts)
                label = 0
                if t_start <= t_fail <= t_end:
                    label = 1
                elif t_start > t_fail:
                    label = 2
                if label == 1 and prev_label == 0:
                    first_one_idx = clip_idx
                elif label == 2 and prev_label == 1:
                    first_two_idx = clip_idx
                    break
                prev_label = label

            self.video_clips.labels.append(
                [0 for i in range(first_one_idx)] + [1 for i in range(first_one_idx, first_two_idx)] +
                [2 for i in range(first_two_idx, len(vid_clips))])

            #Leaving the part: balance_fails_only (I dunno what this is!!)

        print('\nNumber of CLIPS generated:', self.video_clips.num_clips())


    def __len__(self):
        if self.get_video_wise:
            return len(self.video_clips.labels) - self.start_id
        else:
            return self.video_clips.num_clips()

    def __getitem__(self, idx):
        idx = self.start_id + idx

        if self.get_video_wise:             #TO return all clips of a single video 

            labels = self.video_clips.labels[idx]   #here idx is video_idx
            num_of_clips = len(labels)
            
            num_of_clips_before_this_video = 0
            for l in self.video_clips.labels[:idx]:
                num_of_clips_before_this_video += len(l)

            start_clip_id = num_of_clips_before_this_video
            end_clip_id = num_of_clips_before_this_video + num_of_clips 

            video = []
            for idx in range(start_clip_id, end_clip_id):
                clip, _, _, _  = self.video_clips.get_clip(idx)
                if self.transforms:
                    clip = self.transforms(clip)
                    clip = clip.permute(1,0,2,3)
                video.append(clip.unsqueeze(0))
            video = torch.cat(video, dim=0)
            #labels = torch.cat(labels)

            return video, labels

        else:
            video_idx, clip_idx = self.video_clips.get_clip_location(idx)
            video, audio, info, video_idx = self.video_clips.get_clip(idx)
            video_path = self.video_clips.video_paths[video_idx]
            label = self.video_clips.labels[video_idx][clip_idx]

            if self.transforms is not None:
                video = self.transforms(video)

            video = video.permute(1,0,2,3)

            return video, label

        
        
        #print('Get: idx=',idx,video_idx,clip_idx,'/',len(self.video_clips.clips[video_idx]), self.video_clips.labels[video_idx][clip_idx])


def get_video_loader(datapath, annotations_path, batch_size=1, val=False, get_video_wise=False, 
    cached_all_train_data_name='cached_all_train_data.pt', cached_valid_train_data_name='cached_valid_train_data.pt', 
    cached_all_val_data_name='cached_all_val_data.pt', cached_valid_val_data_name='cached_valid_val_data.pt',
    num_workers=0, fps=None, frames_per_clip=None, step_between_clips=None, start_id=0):
    
    dataset = Dataset(datapath=datapath, annotations_path=annotations_path, transforms=train_transform, get_video_wise=get_video_wise, val=val, 
        cached_all_train_data_name=cached_all_train_data_name, cached_valid_train_data_name=cached_valid_train_data_name,
        cached_all_val_data_name=cached_all_val_data_name, cached_valid_val_data_name=cached_valid_val_data_name,
        fps=fps, frames_per_clip=frames_per_clip, step_between_clips=step_between_clips, start_id=start_id)

    sampler = None
    
    if get_video_wise:
        batch_size = 1

    return data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)



if __name__ == '__main__':
    '''
    annotations_path = '/home/cvpr/Documents/Dipayan/DD/PATH/TO/annotations/transition_times.json'
    datapath = '/home/cvpr/Documents/Dipayan/DD/PATH/TO/datasets'
    
    train_loader = get_video_loader(datapath=datapath, annotations_path=annotations_path, get_video_wise=True, val=False)

    for x,y in train_loader:
        print(x.size(), torch.cat(y))
        break
    '''
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('\nUsing : ',device)
    
    annotations_path = '/home/cvpr/Documents/Dipayan/DD/PATH/TO/annotations/transition_times.json'
    datapath = '/home/cvpr/Documents/Dipayan/DD/PATH/TO/datasets'
    cnn_checkpoint_path = 'saved_models/auto_encoder_512_completeData_stateDict.pt'

    #Dataloaders : 16fps, fpc = 16 frames, sbc = 4 frames
    fps = 16
    frames_per_clip = 16
    step_between_clips = 4
    train_start_id, val_start_id = 0,0
    train_loader = get_video_loader(datapath=datapath, annotations_path=annotations_path, val=False, get_video_wise=True, cached_all_train_data_name='112_112_all_train_data.pt', cached_valid_train_data_name='112_112_valid_train_data.pt', num_workers=0, fps=fps, frames_per_clip=frames_per_clip, step_between_clips=step_between_clips/fps, start_id=train_start_id)
    val_loader = get_video_loader(datapath=datapath, annotations_path=annotations_path, val=True, get_video_wise=True, cached_all_val_data_name='112_112_all_val_data.pt', cached_valid_val_data_name='112_112_valid_val_data.pt', num_workers=0, fps=fps, frames_per_clip=frames_per_clip, step_between_clips=step_between_clips/fps, start_id=val_start_id)           
    
