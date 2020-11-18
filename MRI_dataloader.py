import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset
import os
import numpy as np
from tqdm import tqdm

class MRI_LR_video_Dataset(Dataset):
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.filenames = os.listdir(self.filepath)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        x,y = np.load(os.path.join(self.filepath, self.filenames[idx]), allow_pickle=True)
        return torch.from_numpy(x),torch.from_numpy(y)

def MRI_LR_video_dataloader(filepath, batch_size=1, num_workers=0):
    dataset = MRI_LR_video_Dataset(filepath)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


if __name__ == '__main__':

    #Checking numpy dataloader
    numpy_train_path = '/home/cvpr/Documents/Dipayan/DD/MRI_data/train/numpy_cubic_data/'
    numpy_val_path = '/home/cvpr/Documents/Dipayan/DD/MRI_data/val/numpy_cubic_data/'

    train_loader = numpy_MRI_LR_dataloader(numpy_train_path, batch_size=2)
    val_loader = numpy_MRI_LR_dataloader(numpy_val_path, batch_size=1)
    for x,y in train_loader:
        print(x.size(), type(x), y.size(), type(y), y)
        break


    

    



    
