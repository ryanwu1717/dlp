import numpy as np
import math


from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


import torch.nn as nn
class Generator(nn.Module):
    def __init__(self, noise_size):
        super(Generator, self).__init__()
        
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(noise_size, 512, 4, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 5, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 3, 1, 1, 1, bias=False),
            nn.Tanh()
            #nn.Sigmoid()
        )  
        self.noise_size = noise_size
        
    def forward(self, x):
        return self.gen(x)

    '''
    def forwardQ(self, x):
        x = x.view(x.size(0), -1)
        return self.Qlayer(x).squeeze()        
    '''

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(4, 32, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
        )

        self.toinput = nn.Linear(24, 64 * 64, bias=True)

    def forward(self, x, l):
        l = self.toinput(l).view(-1, 1, 64, 64)
        x = torch.cat([x, l], 1)
        return self.main(x).view(-1, 1)



'''
nn.ReLU(),
nn.Linear(50, 24, bias=True),
'''