"""
Trains MADE on Binarized MNIST, which can be downloaded here:
https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz
"""
import argparse

import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from made import MADE

# ------------------------------------------------------------------------------
def run_epoch(split, upto=None):
    torch.set_grad_enabled(split=='train') # enable/disable grad for efficiency of forwarding test batches
    model.train() if split == 'train' else model.eval()
    nsamples = 1 if split == 'train' else args.samples
    x = xtr if split == 'train' else xte
    N,D = x.size()
    B = 100 # batch size
    nsteps = N//B if upto is None else min(N//B, upto)
    lossfs = []
    for step in range(nsteps):

        # fetch the next batch of data
        xb = Variable(x[step*B:step*B+B])

        # get the logits, potentially run the same batch a number of times, resampling each time
        xbhat = torch.zeros_like(xb)
        for s in range(nsamples):
            # perform order/connectivity-agnostic training by resampling the masks
            if step % args.resample_every == 0 or split == 'test': # if in test, cycle masks every time
                model.update_masks()
            # forward the model
            xbhat += model(xb)
        xbhat /= nsamples

        # evaluate the binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(xbhat, xb, size_average=False) / B
        lossf = loss.data.item()
        lossfs.append(lossf)

        # backward/update
        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()

    print("%s epoch average loss: %f" % (split, np.mean(lossfs)))


def run_epoch2(split, upto=None):
    torch.set_grad_enabled(split=='train') # enable/disable grad for efficiency of forwarding test batches
    model.train() if split == 'train' else model.eval()
    nsamples = 1 if split == 'train' else args.samples
    x = xtr if split == 'train' else xte
    N,D = x.size()
    B = 100 # batch size
    nsteps = N//B if upto is None else min(N//B, upto)
    lossfs = []
    for step in range(nsteps):

        # fetch the next batch of data
        xb = Variable(x[step*B:step*B+B])

        # get the logits, potentially run the same batch a number of times, resampling each time
        xbhat = torch.zeros_like(xb)
        for s in range(nsamples):
            # perform order/connectivity-agnostic training by resampling the masks
            if step % args.resample_every == 0 or split == 'test': # if in test, cycle masks every time
                #print("!! nsamples = ", nsamples)
                model.update_masks()
            # forward the model
            xbhat += model(xb)
        xbhat /= nsamples

        #get a sample
        samp = xbhat[0,:]
        samp_matrix = (samp.data).cpu().numpy()
        print("samp_matrix.shape: ", samp_matrix.shape)
        img = samp_matrix.reshape(3,32,32)
        img = np.transpose(img, (1, 2, 0))
        plt.imsave("output_sample_cifar.png", img)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', required=True, type=str, help="Path to cifar-10-python.npz")
    parser.add_argument('-q', '--hiddens', type=str, default='500', help="Comma separated sizes for hidden layers, e.g. 500, or 500,500")
    parser.add_argument('-n', '--num-masks', type=int, default=1, help="Number of orderings for order/connection-agnostic training")
    parser.add_argument('-r', '--resample-every', type=int, default=20, help="For efficiency we can choose to resample orders/masks only once every this many steps")
    parser.add_argument('-s', '--samples', type=int, default=1, help="How many samples of connectivity/masks to average logits over during inference")
    args = parser.parse_args()
    # --------------------------------------------------------------------------

    # reproducibility is good
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # load the dataset
    print("loading cifar from cifar data folder")
    x = []
    l = []
    path = "cifar-10-batches-py/"
    for i in range(1, 6):
        f = open(path + 'data_batch_' + str(i), 'rb')
        dict = pickle.load(f, encoding='latin1')
        x.append(dict['data'])
        l.append(dict['labels'])
        f.close()
    xtr = np.concatenate(x, axis=0)
    #print("xtr.shape: ", xtr.shape)
    l = np.concatenate(l, axis=0)
    # load test batch
    f = open(path + 'test_batch', 'rb')
    dict = pickle.load(f, encoding='latin1')
    xte = dict['data']
    #print("xte.shape: ", xte.shape)
    l = np.array(dict['labels'])
    f.close()

    #construct torch cuda tensors
    xtr = torch.from_numpy(xtr).cuda()
    xte = torch.from_numpy(xte).cuda()
    xtr = xtr.type(torch.cuda.FloatTensor)
    xte = xte.type(torch.cuda.FloatTensor)

    # construct model and ship to GPU
    hidden_list = list(map(int, args.hiddens.split(',')))
    model = MADE(xtr.size(1), hidden_list, xtr.size(1), num_masks=args.num_masks)
    print("number of model parameters:",sum([np.prod(p.size()) for p in model.parameters()]))
    model.cuda()

    # set up the optimizer
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=45, gamma=0.1)

    # start the training
    for epoch in range(100):
        print("epoch %d" % (epoch, ))
        scheduler.step(epoch)
        run_epoch('test', upto=5) # run only a few batches for approximate test accuracy
        run_epoch('train')

    print("optimization done. full test set eval:")
    run_epoch2('test')
