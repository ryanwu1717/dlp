#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dataloader
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce


# In[2]:


class EEGNet(nn.Module):
    # build the EEGNet which show in spec 
    def __init__(self, activation = None, dropout=0.25):
        super(EEGNet, self).__init__()
        #three activation function to use
        activation_dict = nn.ModuleDict([['ELU', nn.ELU(alpha = 1.0)], ['ReLU', nn.ReLU()], ['LeakyReLU', nn.LeakyReLU()]])

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = (1, 51), stride = (1,1), padding = (0,25), bias = False ),
            nn.BatchNorm2d(16, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        )
        
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = (2,1), stride = (1,1), groups = 16, bias = False ),
            nn.BatchNorm2d(32, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            activation_dict[activation],
            nn.AvgPool2d(kernel_size = (1, 4), stride = (1, 4), padding = 0),
            nn.Dropout(p = dropout)
        )
        
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size = (1, 15), stride = (1,1), padding = (0, 7), bias = False
            ),
            nn.BatchNorm2d(32, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            #選擇 activation function
            activation_dict[activation],
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p = dropout)
        )
        
        self.classify = nn.Sequential(
            nn.Linear(736, out_features = 2, bias = True)
        )
        
     
    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        # flatten
        x = x.view(x.size(0), -1)
        output = self.classify(x)
        return output


# In[3]:




class DeepConvNet(nn.Module):
    # build the DeepConvNet which show in spec
    def __init__(self, activation=None, deepconv=[25,50,100,200], dropout=0.5):
        super(DeepConvNet, self).__init__()
        activation_dict = nn.ModuleDict([['ELU', nn.ELU(alpha = 1.0)], ['ReLU', nn.ReLU()], ['LeakyReLU', nn.LeakyReLU()]])
        
        self.deepconv = deepconv
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size = (1, 5), stride = (1, 1), bias = False),
            nn.Conv2d(25, 25, kernel_size = (2, 1), stride = (1, 1), bias = False),
            nn.BatchNorm2d(25, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            activation_dict[activation],
            nn.MaxPool2d(kernel_size = (1, 2)),
            nn.Dropout(p = 0.5)
        )
        
        for idx in range(1, len(deepconv)):
            setattr(self, 'conv'+str(idx), nn.Sequential(
                nn.Conv2d(deepconv[idx-1], deepconv[idx], kernel_size=(1,5),stride=(1,1), padding=(0,0), bias=False),
                nn.BatchNorm2d(deepconv[idx], eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
                activation_dict[activation],
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=dropout)
            ))
        

        self.classify = nn.Sequential(
            nn.Linear(8600, 2, bias=True),
        )
    
    def forward(self, x):
        for i in range(len(self.deepconv)):
            x = getattr(self, 'conv'+str(i))(x)
        # flatten
        x = x.view(x.size(0), -1)
        output = self.classify(x)
        return output


# In[4]:


def draw(lines,model,title,accline=0):


    plt.figure()

    for e in range(len(lines)):

        plt.plot(lines[e]['epoch'], lines[e][title], label = lines[e]['name'])

    plt.xlabel('epoch')
    plt.ylabel(title)

    plt.title(title)
    plt.legend()
    
    #make a 0.87 acc
    if accline != 0:
            plt.hlines(accline, 1, len(lines[0][title]), linestyles='--', colors=(0,0,0,0.5))
    plt.savefig(str(model) +"_" +str(title) + ".png")
    plt.show()
    plt.close()


# In[ ]:


if __name__ == '__main__':

   
    # hyperparamaters
    batch_size = 256
    lr = 7e-4
    epochs = 1000
    
    #check if using gpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print("Computer is  using {} to do this lab.  ".format(device))
    
    models = ["EEGNet", "DeepConvNet"] 
    activations = ["ReLU", "ELU", "LeakyReLU"]
    max_dict={"ReLU":[],"ELU":[],"LeakyReLU":[]}
    
    #data prepare
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_data = torch.tensor(train_data, dtype = torch.float).to(device)
    train_label = torch.tensor(train_label, dtype = torch.long).to(device)
    test_data = torch.tensor(test_data, dtype = torch.float).to(device)
    test_label = torch.tensor(test_label, dtype = torch.long).to(device)
    train_Dataset = Data.TensorDataset(train_data, train_label)
    train_loader = Data.DataLoader(train_Dataset, batch_size = batch_size, shuffle = True)
    
    # two model
    for model in models:
        line_num = 0
        accuracy = []
        losses = []
            
        #three activation function 
        for activation in activations:
            #create model
            if model == "DeepConvNet":
                Net = DeepConvNet(activation)
            elif model == "EEGNet":
                Net = EEGNet(activation)

            print(f'model: {model}, activation: {activation}\n')
            Net.to(device)
            
            max_acc = 0
            
            Loss = nn.CrossEntropyLoss()
            name = model +"_"+ activation+"_"

            optimizer = optim.Adam(Net.parameters(), lr = lr)
            accuracy.append({'epoch': [], 'accuracy': [], 'name': name + "train"})
            accuracy.append({'epoch': [], 'accuracy': [], 'name': name + "test"})

            losses.append({'epoch': [], 'loss': [], 'name': name + "train"})
            
            #start training
            for epoch in range(1,epochs+1):

                #train
                running_loss = 0.0
                train_correct = 0
                Net.train()
                for (data, label) in train_loader:      
                    optimizer.zero_grad()
                    output = Net(data)
                    predict = output.data.max(1)[1]
                    train_correct += predict.eq(label).cpu().sum().item() #check predict
                    loss = Loss(output, label)
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                
                
                accuracy[line_num]['epoch'].append(epoch)
                accuracy[line_num]['accuracy'].append(train_correct/len(train_data))

                losses[int(line_num/2)]['epoch'].append(epoch)
                losses[int(line_num/2)]['loss'].append(running_loss)

                #test
                Net.eval()
                with torch.no_grad():
                    test_output = Net(test_data)
                predict = test_output.data.max(1)[1]
                test_correct = predict.eq(test_label).cpu().sum().item() #check predict
                
                #get the max accuracy
                max_acc = max(max_acc,test_correct/len(test_data))

                accuracy[line_num+1]['epoch'].append(epoch)
                accuracy[line_num+1]['accuracy'].append(test_correct/len(test_data))

                #show the accuracy and loss of and check it is still running
                if epoch % 100 == 0 :
                    print(f'\nepoch: {epoch}\ntest accuracy: {test_correct/len(test_data)}')
                    print(f'loss: {running_loss}')

            line_num += 2
            max_dict[activation].append(max_acc)
            print(f'\nmax test accuracy: {max_acc}\n')

        #draw the compare curves
        draw(accuracy,model,"accuracy",0.87)
        draw(losses,model,"loss")
        
    #show all max accuracy by DataFrame
    df = pd.DataFrame(max_dict,index=models)    
    print(df)
    print("END")


# In[ ]:





# In[ ]:




