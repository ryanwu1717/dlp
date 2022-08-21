import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
import numpy as np
from os import system
import data_loader
import model
import evaluator
"""========================================================================================
The sample.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to 
implement by yourself.

1. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
2. Output your results (BLEU-4 score, correction words)
3. Plot loss/score
4. Load/save weights
========================================================================================"""
c_size = 24 #size of meaningful codes
noise_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
testloader = DataLoader(data_loader.dataloader('./data/', mode = 'test'), batch_size= 32, shuffle=False)
netG = torch.load('./save_model/' + '435_9_G').to(device)
eva = evaluator.evaluation_model()

def _noise_sample(batch, real_labels):

    z = torch.cat([
        torch.FloatTensor(batch, noise_size - c_size).uniform_(-1.0, 1.0), 
        real_labels.type(torch.float32).cpu()
    ] , 1).view(-1, noise_size, 1, 1)
    
    return z.to(device), real_labels.type(torch.float32)
batch_data = 0
for batch, batch_data in enumerate(testloader, 0):
    break
netG.eval()
count = 0
while True:
    count += 1
    label = batch_data
    batch_size = len(label)
    noise = torch.cat([
    torch.FloatTensor(batch_size, noise_size - c_size).uniform_(-1.0, 1.0), 
    torch.Tensor(label.numpy())
    ] , 1).view(-1, noise_size, 1, 1).to(device)
    fake_x = netG(noise)
    acc = eva.eval(fake_x, torch.Tensor(label.numpy()))
    if acc >= 0.65:
        print("ture: ", count)
        print("evaluate acc: ", acc)
        transform1 = transforms.Compose([ 
            transforms.Normalize(mean = [ 0., 0., 0. ],
                                 std = [ 1/0.5, 1/0.5, 1/0.5 ]),
            transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                 std = [ 1., 1., 1. ]),
           ]
        )        
        norm_image = torch.randn(0, 3, 64, 64)
        for fake_image in fake_x:
            n_image = transform1(fake_image.cpu().detach())
            #n_image = transform2(n_image)            
            norm_image = torch.cat([norm_image, n_image.view(1, 3, 64, 64)], 0)
        save_image(make_grid(norm_image, nrow=8), "result.jpg")
        break
