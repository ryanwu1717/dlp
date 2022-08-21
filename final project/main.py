import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid

import numpy as np
import data_loader
import model
import evaluator
from Visualization import *

version = '50'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#----------Hyper Parameters----------#
Batch_size = 16
Image_size = 64
LR_Discriminator = 2e-4
LR_Generator_G = 1e-3

c_size = 24 #size of meaningful codes
noise_size = 64
Total_epochs = 1

print('device:',device, '  version:', version, ' epoch:', Total_epochs)
dataloader = DataLoader(data_loader.dataloader('./dataset/', mode = 'train'), batch_size=Batch_size, shuffle=True)
testloader = DataLoader(data_loader.dataloader('./dataset/', mode = 'test'), batch_size= 32, shuffle=False)


netG = model.Generator(noise_size).to(device)

netD = model.Discriminator().to(device)

optimizerD = optim.Adam([{'params':netD.main.parameters()}, {'params':netD.toinput.parameters()}], lr=LR_Discriminator, betas=(0.5, 0.999))
optimizerG = optim.Adam([{'params':netG.gen.parameters()}], lr=LR_Generator_G, betas=(0.5, 0.999))

criterion_D = nn.BCELoss()
criterion_L2 = nn.MSELoss()
eva = evaluator.evaluation_model()
def _noise_sample(batch, real_labels):
    '''
    class_num = np.random.randint(3, size=batch)
    c = np.zeros((batch, c_size))

    for i, e in enumerate(class_num, 0):
        idx = np.random.randint(c_size, size=e)
        c[i, idx] = 1.0
    '''
    z = torch.cat([
        torch.FloatTensor(batch, noise_size - c_size).uniform_(-1.0, 1.0), 
        real_labels.type(torch.float32).cpu()
    ] , 1).view(-1, noise_size, 1, 1)
    
    return z.to(device), real_labels.type(torch.float32)

"""
Plot the loss curve
"""
plot_loss_generator = []
plot_loss_discriminator = []
plot_loss_Q = []
plot_test_acc = []
"""
Plot Probability
"""
plot_D_real_data = []
plot_D_fake_data_before_updating_G = []
plot_D_fake_data_after_updating_G = []

for epoch in range(Total_epochs + 1):
    netD.train()
    netG.train()
    total_G_loss = 0 
    total_D_loss = 0
    total_Q_loss = 0
    for batch, batch_data in enumerate(dataloader, 0):
        return
        ############################
        # (1) Update D network:
        ###########################
        #train with real
        optimizerD.zero_grad()

        x, real_label = batch_data
        real_label = real_label.to(device).type(torch.float32)
        bs = x.size(0)
        real_x = x.to(device)
        label = torch.FloatTensor(bs, 1).to(device)
        label.data.fill_(1.0)
        real_out = netD(real_x, real_label)
        D_x = real_out.mean().item()
        loss_D_real = criterion_D(real_out, label)
        loss_D_real.backward()

        # train with fake
        z, idx = _noise_sample(bs, real_label)
        fake_x = netG(z)
        fake_out = netD(fake_x.detach(), real_label)
        D_G_z1 = fake_out.mean().item()
        label.data.fill_(0)
        loss_D_fake = criterion_D(fake_out, label)
        loss_D_fake.backward()
        loss_D = loss_D_real + loss_D_fake
        total_D_loss += loss_D.item()
        optimizerD.step()
        
        ############################
        # (2) Update G network:
        ###########################
        optimizerG.zero_grad()

        fe_out = netD(fake_x, real_label)
        label.data.fill_(1.0)
        D_G_z2 = fe_out.mean().item()
        loss_L2 = criterion_L2(fake_x, real_x) # l2 loss
        loss_G_reconstruct = criterion_D(fe_out, label)
        loss_G = loss_G_reconstruct  + loss_L2
        total_G_loss += loss_G.item()
        loss_G.backward()
        optimizerG.step()
        
        if batch % 1125 == 0 and batch != 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f Q: %.4f'
              % (epoch, Total_epochs, batch, len(dataloader),
                 total_D_loss / (bs*Batch_size) , total_G_loss / (bs*Batch_size), D_x, D_G_z1, D_G_z2, total_Q_loss / (bs * Batch_size)))

            plot_loss_generator.append(total_G_loss / (bs*Batch_size))
            plot_loss_discriminator.append(total_D_loss / (bs*Batch_size))

    # test
    netG.eval()
    for batch, batch_data in enumerate(testloader, 0):
        label = batch_data
        batch_size = len(label)
        z = torch.cat([
        torch.FloatTensor(batch_size, noise_size - c_size).uniform_(-1.0, 1.0), 
        torch.Tensor(label.numpy())
        ] , 1).view(-1, noise_size, 1, 1).to(device)
        fake_x = netG(z)
        acc = eva.eval(fake_x, torch.Tensor(label.numpy()))
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

        save_image(make_grid(norm_image, nrow=8), "temp.jpg")
        plot_test_acc.append(acc)
        print("acc: ", acc)

    if epoch % 3 == 0:
        Plot_loss(plot_loss_generator, plot_loss_discriminator, None, version, epoch)
        Plot_acc(plot_test_acc, version, epoch)
        torch.save(netD, "./save_model/" + str(epoch) + '_' + version + '_D')
        torch.save(netG, "./save_model/" + str(epoch) + '_' + version + '_G')
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
        save_image(make_grid(norm_image, nrow=8),"./figure/"+ version+ '_'+ str(epoch) + ".jpg")
