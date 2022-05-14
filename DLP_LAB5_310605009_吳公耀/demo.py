import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import bair_robot_pushing_dataset
# import dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, plot_pred, plot_rec, finn_eval_seq, pred, mse_metric,save_gif_with_text,save_tensors_image
import Visualization

import os
import gc

gc.collect()
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
torch.backends.cudnn.benchmark = True
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
    parser.add_argument('--model_path', default='./logs/fp/rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000/model.pth', help='path to model')
    parser.add_argument('--log_dir', default='./logs/fp/rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000', help='directory to save generations to')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--num_threads', type=int, default=0, help='number of data loading threads')
    parser.add_argument('--nsample', type=int, default=3, help='number of samples')
    parser.add_argument('--N', type=int, default=256, help='number of samples')
    parser.add_argument('--cuda', default=True, action='store_true')  
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    args = parser.parse_args()
    return args
# --------- eval funtions ------------------------------------

def make_gifs(x,cond, idx, modules,args, device):
    x = x.to(device)
    cond = cond.to(device)
    # get approx posterior sample
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0].type(torch.cuda.FloatTensor)
    for i in range(1, args.n_past + args.n_future):
        h = modules['encoder'](x_in)
        h_target = modules['encoder'](x[i].type(torch.cuda.FloatTensor))[0].detach()
        if args.last_frame_skip or i < args.n_past:	
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        _, z_t, _= modules['posterior'](h_target) # take the mean
        if i < args.n_past:
            modules['frame_predictor'](torch.cat([h, z_t], 1),cond[i-1]) 
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            h_pred = modules['frame_predictor'](torch.cat([h, z_t], 1),cond[i-1]).detach()
            x_in = modules['decoder']([h_pred, skip]).detach()
            posterior_gen.append(x_in)
  

    nsample = 3
    ssim = np.zeros((args.batch_size, nsample, args.n_future))
    psnr = np.zeros((args.batch_size, nsample, args.n_future))

    all_gen = []
    for s in range(nsample):

        gen_seq = []
        gt_seq = []
        modules['frame_predictor'].hidden =  modules['frame_predictor'].init_hidden()
        modules['posterior'].hidden = modules['posterior'].init_hidden()
        x_in = x[0]
        all_gen.append([])
        all_gen[s].append(x_in)
        for i in range(1,args.n_past + args.n_future):
            h =  modules['encoder'](x_in)
            if args.last_frame_skip or i < args.n_past:	
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < args.n_past:
                h_target =  modules['encoder'](x[i])[0].detach()
                _, z_t, _ =  modules['posterior'](h_target)
            else:
                z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_()
            if i < args.n_past:
                modules['frame_predictor'](torch.cat([h, z_t], 1),cond[i-1])
                x_in = x[i]
                all_gen[s].append(x_in)
            else:
                h = modules['frame_predictor'](torch.cat([h, z_t], 1),cond[i-1]).detach()
                x_in = modules['decoder']([h, skip]).detach()
                gen_seq.append(x_in)
                gt_seq.append(x[i])
                all_gen[s].append(x_in)
        _, ssim[:, s, :], psnr[:, s, :] = finn_eval_seq(gt_seq, gen_seq)
        

    to_plot = []
    row = [] 
    for t in range(args.n_past+args.n_future):
        row.append(x[t][0])
    to_plot.append(row)
    row = []
    for t in range(args.n_past+args.n_future):
        row.append(all_gen[0][t][0]) 
    to_plot.append(row)
    fname = 'demo.png' 
    save_tensors_image(fname,to_plot)
    ###### psnr ######
    for i in range(args.batch_size):
        gifs = [ [] for t in range(args.n_past + args.n_future) ]
        text = [ [] for t in range(args.n_past + args.n_future) ]
        mean_ssim = np.mean(psnr[i], 1)
        ordered = np.argsort(mean_ssim)
        rand_sidx = [np.random.randint(nsample) for s in range(3)]
        for t in range(args.n_past + args.n_future):
            # gt 
            gifs[t].append(add_border(x[t][i], 'green'))
            text[t].append('Ground\ntruth')
            #posterior 
            if t < args.n_past:
                color = 'green'
            else:
                color = 'red'
            gifs[t].append(add_border(posterior_gen[t][i], color))
            text[t].append('Approx.\nposterior')
            # best 
            if t < args.n_past:
                color = 'green'
            else:
                color = 'red'
            sidx = ordered[-1]
            gifs[t].append(add_border(all_gen[sidx][t][i], color))
            text[t].append('Best PSNR')
            # random 3
            for s in range(len(rand_sidx)):
                gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                text[t].append('Random\nsample %d' % (s+1))

        fname = 'demo.gif' 
        save_gif_with_text(fname, gifs, text)
        break

def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px

  
def main():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    os.makedirs('%s/demo' % args.log_dir, exist_ok=True)


    args.n_eval = args.n_past+args.n_future
    args.max_step = args.n_eval
 
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype = torch.cuda.FloatTensor
    # ---------------- load the models  ----------------
    tmp = torch.load(args.model_path)
    frame_predictor = tmp['frame_predictor']
    posterior = tmp['posterior']
    #prior = tmp['prior']

    encoder = tmp['encoder']
    decoder = tmp['decoder']
    
    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()
    #prior.eval()

    frame_predictor.batch_size = args.batch_size
    posterior.batch_size = args.batch_size
    #prior.batch_size = args.batch_size
    args.g_dim = tmp['args'].g_dim
    args.z_dim = tmp['args'].z_dim
    # --------- transfer to gpu ------------------------------------
    frame_predictor.cuda()
    posterior.cuda()
    #prior.cuda()
    encoder.cuda()
    decoder.cuda()
    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
        #'prior': prior,
    }
    # ---------------- set the options ----------------
    
    args.last_frame_skip = tmp['args'].last_frame_skip
    args.channels = 1
    args.image_width = 64
    
    # --------- load a dataset ------------------------------------
    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    test_data = bair_robot_pushing_dataset(args, 'test')
    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    train_iterator = iter(train_loader)
    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    test_iterator = iter(test_loader)
    for _ in range(len(test_data) // args.batch_size):

        try:
            test_seq, test_cond = next(test_iterator)
        except StopIteration:
            test_iterator = iter(test_loader)
            test_seq, test_cond = next(test_iterator)
            
        newseq = test_seq.permute(1,0,2,3,4)
        newcond = test_cond.permute(1,0,2)

        pred_seq = pred(newseq, newcond, modules, args, device,999)
        
        _, _, psnr = finn_eval_seq(newseq[args.n_past:], pred_seq[args.n_past:])
      


    ave_psnr = np.mean(np.concatenate(psnr))
    print('psnr: {}'.format(ave_psnr))


    try:
        seq, cond = next(train_iterator)
    except StopIteration:
        train_iterator = iter(train_loader)
        seq, cond = next(train_iterator)

    newseq = seq.permute(1,0,2,3,4)
    newcond = cond.permute(1,0,2)
    
    plot_pred(newseq,newcond, modules, 999, args, device)
    make_gifs(newseq, newcond, 'demo',modules, args, device)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()