import pandas as pd
from torch.utils import data
import numpy as np
import json
from PIL import Image
import torchvision.transforms as transforms

class dataloader(data.Dataset):
    def __init__(self, root = './data/', mode = 'train'):
        input_file = open('objects' + '.json')
        json_dic = json.load(input_file)
        self.dic = json_dic
        self.img_name = []
        self.label = []
        if mode == 'train':
            input_file = open(mode + '.json')
            json_array = json.load(input_file)
            for e in json_array:
                self.img_name.append(e)
                label = np.asarray([0] * 24) 
                for i in json_array[e]:
                  label[self.dic[i]] = 1
                self.label.append(label)
        else :
            input_file = open(mode + '.json')
            json_array = json.load(input_file)
            for e in json_array:
                label = np.asarray([0] * 24) 
                for i in e:
                  label[self.dic[i]] = 1.0
                self.label.append(label)

        #self.img_name = self.img_name[:32]
        #self.label = self.label[:32]
        self.root = root
        self.mode = mode
        print(f"{mode} found %d images..." % (len(self.label)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.label)

    def __getitem__(self, index):
        """something you should implement here"""

        if self.mode == 'train':
            path = self.root + self.img_name[ index ] 
            label = self.label[ index ]

            image = Image.open(path).convert('RGB') 
            transform = transforms.Compose([transforms.RandomCrop(240),
                transforms.RandomHorizontalFlip() ,
                transforms.Resize((64,64)), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            #mean=[0.485, 0.456, 0.406],
            #                         std=[0.229, 0.224, 0.225]
            img = transform(image)

            return img, label
        else :
            return self.label[ index ]

if __name__ == '__main__':

    dataloader = dataloader('./data/','train')
    print(dataloader.__getitem__(10))

    input_file = open('objects' + '.json')
    json_array = json.load(input_file)
    print(type(json_array))
    for e in json_array:
        print(json_array[e])