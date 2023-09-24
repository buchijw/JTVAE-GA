import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
import wandb

import math, random
import numpy as np
import argparse

import os
import sys
sys.path.append('%s/../fast_jtnn/' % os.path.dirname(os.path.realpath(__file__)))
from mol_tree import Vocab, MolTree
from jtprop_vae import JTPropVAE
from jtnn_enc import JTNNEncoder
from jtmpn import JTMPN
from mpn import MPN
from nnutils import create_var
from datautils import PropMolTreeFolder, PairTreeFolder, PropMolTreeDataset
# from fast_jtnn import *
import rdkit
from timeit import default_timer as timer
import datetime

from buchiutils.train_utils import save_train_hyper_csv,save_train_progress

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_epoch', type=int, default=0)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--step_beta', type=float, default=0.002)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=40000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=40000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=5000)

args = parser.parse_args()
print(args)

# Save hyperparameteres
if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)
hyper_dict = vars(args)
now = datetime.datetime.now()
dt_nowstr = now.strftime('%Y-%m-%d_%H.%M.%S')
save_train_hyper_csv(hyper_dict,'%s/hyperparam_%s.csv'%(args.save_dir,dt_nowstr))

# Init wandb
wandb.init(project="JTVAE-with-prop",config=hyper_dict)

# Prepare model
vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

model = JTPropVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda()
print(model)

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

if args.load_epoch > 0:
    model.load_state_dict(torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch)))
    last_timer = timer()

print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

# Init states
wandb.watch(model,log='all',log_freq=10000)
total_step = args.load_epoch
beta = args.beta
meters = np.zeros(6)
start_timer = timer()

# Training loop
for epoch in range(args.epoch):
    loader = PropMolTreeFolder(args.train, vocab, args.batch_size, num_workers=4)
    epoch_timer = timer()
    for batch in loader:
        total_step += 1
        try:
            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc, pacc = model(batch, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
        except Exception as e:
            print(e)
            continue

        loss_np = loss.item()
        # print(loss_np)
        meters = meters + np.array([loss_np, kl_div, wacc * 100, tacc * 100, sacc * 100, pacc])
        

        if total_step % args.print_iter == 0:
            meters /= args.print_iter
            stop_timer = timer()
            if total_step == args.print_iter:
                iter_time = stop_timer - epoch_timer
            else:
                iter_time = stop_timer - last_timer
            print("[%d] Beta: %.3f, Loss: %.5f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, Prop: %.4f, PNorm: %.2f, GNorm: %.2f, Iter Time: %.2f, Epoch Time: %s, Elap Time: %s" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model), grad_norm(model),iter_time,str(datetime.timedelta(seconds=stop_timer - epoch_timer)), str(datetime.timedelta(seconds=stop_timer - start_timer))))
            progress_info = {'total_step':total_step, 'epoch': epoch, 'Beta':beta, 'Loss':meters[0], 'KL':meters[1], 'Word':meters[2], 'Topo': meters[3], 'Assm':meters[4], 'Prop':meters[5], 'PNorm': param_norm(model), 'GNorm':grad_norm(model), 'iter_time':iter_time, 'epoch_time':datetime.timedelta(seconds=stop_timer - epoch_timer),'elap_time':datetime.timedelta(seconds=stop_timer - start_timer),'lr':scheduler.get_last_lr()[0]}
            save_train_progress(progress_info, '%s/train_progress_%s.csv'%(args.save_dir,dt_nowstr), reset= bool(total_step == args.print_iter))
            wandb.log({'epoch': epoch, 'Beta':beta, 'Loss':meters[0], 'KL':meters[1], 'Word':meters[2], 'Topo': meters[3], 'Assm':meters[4], 'Prop':meters[5], 'PNorm': param_norm(model), 'GNorm':grad_norm(model),'lr':scheduler.get_last_lr()[0]},step = total_step)
            sys.stdout.flush()
            meters *= 0
            last_timer = timer()

        if total_step % args.save_iter == 0:
            torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(total_step))

        if total_step % args.anneal_iter == 0:
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])

        if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
            beta = min(args.max_beta, beta + args.step_beta)

wandb.finish()