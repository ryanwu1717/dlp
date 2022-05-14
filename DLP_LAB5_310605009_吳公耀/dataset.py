import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage.io import imread
import pandas as pd

default_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        self.data_dir = ''
        self.istrain = False
        self.d = 0
        self.seq_len = 12
        self.seed_is_set = False
        self.path = ''
        
        if mode == 'train':
            self.data_dir = 'data/processed_data/train/'
            self.istrain = True
        elif mode == 'test':
            self.data_dir = 'data/processed_data/test/'
        elif mode == 'validate':
            self.data_dir = 'data/processed_data/validate/'
        self.dirs = []      #put all sequence dirs in here
        for dir1 in os.listdir(self.data_dir):
            for dir2 in os.listdir('%s/%s' %(self.data_dir,dir1)):
                self.dirs.append('%s/%s/%s' %(self.data_dir,dir1,dir2))


    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            
    def __len__(self):
        return len( self.dirs)
        
    def get_seq(self,index):
        #get sequence by index
        d = self.dirs[index]
        self.path = d
        image_seq = []
        #get the file start from 0
        for i in range(self.seq_len):
            fname = '%s/%d.png' % (d, i)
            image = Image.open(fname).convert('RGB')
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(image).reshape(1, 3, 64, 64)
            image_seq.append(img)

        image_seq = np.concatenate(image_seq, axis=0)
        return image_seq.astype(np.float32)

    
    def get_csv(self,index):
        d = self.dirs[index]
        self.path = d
        cond = []
        #get condition by index
        action_path = '%s/actions.csv' %(self.path)
        action = pd.read_csv(action_path)
        action_tensor = torch.tensor(action.values.astype(np.float32))

        position_path = '%s/endeffector_positions.csv' %(self.path)
        position = pd.read_csv(position_path)
        position_tensor = torch.tensor(position.values.astype(np.float32))

        cond = torch.cat((action_tensor,position_tensor), -1)
        cond_split = torch.split(cond,self.seq_len)
        for i in cond_split:
            return i
        return cond

    
    def __getitem__(self, index):
        self.set_seed(index)
        seq = self.get_seq(index)
        cond =  self.get_csv(index)
        return seq, cond



            