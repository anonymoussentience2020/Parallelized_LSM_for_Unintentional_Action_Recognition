import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import time
import os

from MRI_txt_dataloader import MRI_LR_video_dataloader
print('Path example : home/username/Downloads/Parallelized_LSM_for_Unintentional_Action_Recognition')
base_path = input('Enter the absolute path to the PLSM folder, including the name of the PLSM folder (like above example):')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class CNNModel(nn.Module):
    def __init__(self, input_channel):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(input_channel, 64)
        self.fc_2 = nn.Linear(64,3)
        self.softmax = nn.Softmax(dim=1)
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0, stride=3),
        nn.Dropout(p=0.3),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2))
        )
        return conv_layer
    
    def forward(self, x):

        x = self.conv_layer1(x)
        x = torch.reshape(x, (x.size(0),-1))
        
        x = self.fc_2(x)
        x = self.softmax(x)

        return x

def normalize_data(x, mean, std):
    return (x-mean)/std

def nn_accuracy(pred, target):
    pred_ = torch.argmax(pred, dim=1)
    score = sum((pred_==target).float()) 

    return score / len(pred_)

def get_dataset_stats(dataloader):
    total, num_elements = 0, 0
    labels = {0:0, 1:0, 2:0}

    for x,y in tqdm(dataloader):
        total += torch.sum(x)
        num_elements += x.size(0)*x.size(1)*x.size(2)*x.size(3)*x.size(4)
    mean = total/num_elements

    for x,y in tqdm(dataloader):
        total += torch.sum((x-mean)**2)

        labels[y.item()] += 1

    std = torch.sqrt(total/num_elements)

    return mean, std, labels

def get_mask(clip_idx=None, last_output=None, mask_element=0):

    if clip_idx is None and last_output is None:    #If epoch starts 
        return torch.tensor([[1.0,1.0, mask_element]])

    elif clip_idx == 0 and last_output is not None: #If new video starts
        return torch.tensor([[1.0,1.0, mask_element]])

    else:
        if last_output == 0:
            return torch.tensor([[1.0,1.0,mask_element]])
        elif last_output == 1:
            return torch.tensor([[mask_element,1.0,1.0]])
        elif last_output == 2:
            return torch.tensor([[mask_element,mask_element,1.0]])

datafiles = ['PLSM_readout_100timesteps']
input_channels=[100]
for file_idx, filename in enumerate(datafiles):
    
    #If problem arises due to base_path, remove the base_path variables from os.path.join() functions in lines 94,95,96
    print('\n\n\n Training for :  '+filename+'_64 kernels')
    train_data_path = os.path.join(base_path,'MRI_data',filename,'train')
    val_data_path = os.path.join(base_path,'MRI_data',filename,'val')
    model_path = os.path.join(base_path,'saved_models')
    
    print('Initiating dataloaders...')
    train_loader = MRI_LR_video_dataloader(filepath=train_data_path, batch_size=1, num_workers=0)
    val_loader = MRI_LR_video_dataloader(filepath=val_data_path, batch_size=1, num_workers=0)

    #dataset stats calculation  
        #print('Calculating dataset stats...')
        #train_mean, train_std, train_class_div = get_dataset_stats(train_loader)
        #val_mean, val_std, val_class_div = get_dataset_stats(val_loader)
        #print('TRAIN: Mean:{} | Std:{} || VAL: Mean:{} | Std:{}'.format(train_mean, train_std, val_mean, val_std))
        #print('Class Div: Train: {} || Val: {}'.format(train_class_div, val_class_div))
        #Toy:
        #TRAIN: Mean:0.002430528635159135 | Std:0.05000391602516174 || VAL: Mean:0.002430765191093087 | Std:0.05000811815261841
        #Class Div: Train: {0: 7076, 1: 1898, 2: 9221} || Val: {0: 3387, 1: 911, 2: 4108}
        #Complete data:
        #TRAIN: Mean:0.002432013163343072 | Std:0.05011998862028122 || VAL: Mean:0.002428024308755994 | Std:0.04998859390616417
        #Class Div: Train: {0: 58215, 1: 15716, 2: 74953} || Val: {0: 44181, 1: 11938, 2: 57224}

    train_mean, train_std, val_mean, val_std = 0.002432013163343072, 0.05011998862028122, 0.002428024308755994, 0.04998859390616417

    #model_save_name = input('Enter model name:')
    model_save_name = filename+'64_kernels'+'.pt'

    cnn_model = CNNModel(input_channel=input_channels[file_idx]).to(device)
    weights = [0.18, 0.67, 0.115]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001) # , weight_decay=0.05)
    lr_schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3, verbose=True)
    batch_size = 64

    print('CNN Model made.')

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    print('Starting Training...')
    print('Validation will be done after every 5 training epochs...')
    print('After every 50 training epochs, a loss-accuracy plot image will be saved in the base folder...')
    
    for epoch in range(500):

            print('\nEpoch:',epoch)

            running_loss = []
            running_acc = []
            pred_list = []
            targets_list = []
            cnn_model.train()

            print('Training...')
            samples = 0
            MRI_CNN_pred_buffer, MRI_CNN_label_buffer = [], []

            val = False
            for x,y in tqdm(train_loader):
                x_, y_ = x[0].to(device), y[0].to(device)
                for clip_idx, clip in enumerate(x_):
                    x, y = clip, y_[clip_idx]
                    x = normalize_data(x, train_mean, train_std)
                    
                    pred_pdf = cnn_model(x.float().to(device).unsqueeze(0))
                    mask = get_mask(clip_idx=clip_idx, last_output=torch.argmax(MRI_CNN_pred_buffer[-1]).item(), mask_element=0) if len(MRI_CNN_pred_buffer)>0 else get_mask(mask_element=0)#torch.min(logit).item()-1)
                    pred = mask.to(device) * pred_pdf

                    MRI_CNN_pred_buffer.append(pred)
                    MRI_CNN_label_buffer.append(y)
                    samples += 1

                    if samples >= batch_size:
                        loss = criterion(torch.cat(MRI_CNN_pred_buffer,dim=0).to(device), torch.tensor(MRI_CNN_label_buffer).to(device).long())
                        
                        running_loss.append(loss.item())
                        running_acc.append(nn_accuracy(torch.cat(MRI_CNN_pred_buffer,dim=0).cpu(), torch.tensor(MRI_CNN_label_buffer).long().cpu()).item())
                        pred_list.append(torch.argmax(torch.cat(MRI_CNN_pred_buffer,dim=0), dim=1).cpu().numpy())
                        targets_list.append(torch.tensor(MRI_CNN_label_buffer).cpu().numpy())

                        if not val:
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()

                        MRI_CNN_label_buffer, MRI_CNN_pred_buffer = [], []
                        samples = 0
                # Video end

            print('Example output:')
            print([torch.argmax(p.detach(),dim=1).item() for p in MRI_CNN_pred_buffer],':Pred')
            print([l.item() for l in MRI_CNN_label_buffer],':GT')
                
            print('Training accuracy:', np.mean(running_acc))
            print('Training loss:',np.mean(running_loss))
            train_loss.append(np.mean(running_loss))
            train_acc.append(np.mean(running_acc))

            pred_list.pop()
            targets_list.pop()
            pred = np.asarray(pred_list).flatten()
            targets = np.asarray(targets_list).flatten()
            assert len(pred) == len(targets)

            print('confusion_matrix:\n', confusion_matrix(targets, pred, labels=[0,1,2]))

            torch.save(cnn_model.state_dict(), os.path.join(model_path, model_save_name))

            if epoch % 5 == 0:
                running_loss = []
                running_acc = []
                pred_list = []
                targets_list = []
                cnn_model.eval()

                print('Validating...')
                samples = 0
                MRI_CNN_pred_buffer, MRI_CNN_label_buffer = [], []

                val = True
                for x,y in tqdm(val_loader):
                    x_, y_ = x[0].to(device), y[0].to(device)
                    for clip_idx, clip in enumerate(x_):
                        x, y = clip, y_[clip_idx]
                        x = normalize_data(x, val_mean, val_std)
                        
                        pred_pdf = cnn_model(x.float().to(device).unsqueeze(0))
                        mask = get_mask(clip_idx=clip_idx, last_output=torch.argmax(MRI_CNN_pred_buffer[-1]).item(), mask_element=0) if len(MRI_CNN_pred_buffer)>0 else get_mask(mask_element=0)#torch.min(logit).item()-1)
                        pred = mask.to(device) * pred_pdf

                        MRI_CNN_pred_buffer.append(pred)
                        MRI_CNN_label_buffer.append(y)
                        samples += 1

                        if samples >= batch_size:
                            loss = criterion(torch.cat(MRI_CNN_pred_buffer,dim=0).to(device), torch.tensor(MRI_CNN_label_buffer).to(device).long())
                            
                            running_loss.append(loss.item())
                            running_acc.append(nn_accuracy(torch.cat(MRI_CNN_pred_buffer,dim=0).cpu(), torch.tensor(MRI_CNN_label_buffer).long().cpu()).item())
                            pred_list.append(torch.argmax(torch.cat(MRI_CNN_pred_buffer,dim=0), dim=1).cpu().numpy())
                            targets_list.append(torch.tensor(MRI_CNN_label_buffer).cpu().numpy())

                            MRI_CNN_label_buffer, MRI_CNN_pred_buffer = [], []
                            samples = 0
                    
                    #Video end
                    
                print('Example output:')
                print([torch.argmax(p.detach(),dim=1).item() for p in MRI_CNN_pred_buffer],':Pred')
                print([l.item() for l in MRI_CNN_label_buffer],':GT')

                print('Valdiation accuracy:', np.mean(running_acc))
                print('Validation loss:',np.mean(running_loss))
                val_loss.append(np.mean(running_loss))
                val_acc.append(np.mean(running_acc))

                pred_list.pop()
                targets_list.pop()
                pred = np.asarray(pred_list).flatten()
                targets = np.asarray(targets_list).flatten()
                assert len(pred) == len(targets)

                print('confusion_matrix:\n', confusion_matrix(targets, pred, labels=[0,1,2]))

                lr_schedular.step(val_acc[-1])


            if epoch % 50 == 0:

                data = [train_loss, val_loss, train_acc, val_acc]
                np.save(os.path.join(filename+'_'+str(epoch)), data)

                plt.subplot(2,2,1)
                plt.title('Train Loss')
                plt.plot(train_loss)
                plt.subplot(2,2,2)
                plt.title('Val Loss')
                plt.plot(val_loss)
                plt.subplot(2,2,3)
                plt.title('Train Accuracy')
                plt.plot(train_acc)
                plt.subplot(2,2,4)
                plt.title('Val Accuracy')
                plt.plot(val_acc)
                
                plt.savefig(os.path.join(filename+'_'+str(epoch)+'.png'))
                plt.clf()
    
