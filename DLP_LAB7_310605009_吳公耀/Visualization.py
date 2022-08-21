import matplotlib.pyplot as plt
import numpy as np

def Plot_loss(G, D, Q, version, epoch = 20):
    import matplotlib.pyplot as plt
    plt.title('infogan Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(len(G)), G, label = 'generator')
    plt.plot(range(len(D)), D, label = 'discriminator')
    #plt.plot(range(len(Q)), Q, label = 'Q')
    plt.legend()
    plt.savefig('./figure/'+ str(version) +"_loss_"+ str(epoch) + ".png")
    plt.close()
def Plot_acc(acc, version, epoch = 20):
    plt.title('acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(range(len(acc)), acc, label = 'acc')
    plt.legend()
    plt.savefig('./figure/'+ str(version) +"_acc_"+ str(epoch) + ".png")
    plt.close()