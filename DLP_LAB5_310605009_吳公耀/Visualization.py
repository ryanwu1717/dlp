import matplotlib.pyplot as plt
import numpy as np

def draw_score(lines, version, epoch = None ): # draw loss curves

    plt.figure()
    plt.plot(lines['epoch'], lines['psnr'], label = 'psnr')



    plt.xlabel('epoch')
    plt.ylabel('score')
    
    plt.title('psnr score')
    plt.legend()
    plt.savefig('./figure/'+ str(version) +"_score_"+ str(epoch) + ".png")
    plt.close()

def draw_loss(line, version, epoch = None ):

    plt.figure()
    plt.plot(line['epoch'], line['CE_loss'], label = 'CE_loss')
    plt.plot(line['epoch'], line['KLD_loss'], label = 'KLD_loss')   
    plt.plot(line['epoch'], line['KLD_weight'], label = 'KLD_weight')
    plt.plot(line['epoch'], line['tfr'], label = 'teacher_force_ratio')     
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    plt.title('loss')
    plt.legend()
    plt.savefig('./figure/'+ str(version) +"_loss_"+ str(epoch) + ".png")
    plt.close()