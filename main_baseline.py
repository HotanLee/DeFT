import os
import sys
import random
import argparse
import numpy as np

import torch
from utils.config import _C as cfg
from utils.lnl_methods import *
from model import *
from sklearn.mixture import GaussianMixture

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default="", help="path to config file")
parser.add_argument("--noise_mode", default=None)
parser.add_argument("--noise_ratio", default=None)
parser.add_argument("--lnl_methods", default=None, choices=['CE', 'ELR', 'SCE', 'GMM'], help="label-noise learning methods")
parser.add_argument("--gpuid", default=None)

args = parser.parse_args()
cfg.defrost()
cfg.merge_from_file(args.cfg)
if args.noise_mode is not None:
    cfg.noise_mode = args.noise_mode
if args.noise_ratio is not None:
    cfg.noise_ratio = float(args.noise_ratio)
if args.lnl_methods is not None:
    cfg.lnl_methods = args.lnl_methods
if args.gpuid is not None:
    cfg.gpuid = int(args.gpuid)

def set_seed():
    torch.cuda.set_device(cfg.gpuid)
    seed = cfg.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

def eval_train(dataloader):    
    model.eval()
    losses = torch.zeros(len(dataloader.dataset)).cuda().half()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    with torch.no_grad():
        for batch_idx, (inputs, targets,_, index) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs, return_sim=cfg.is_coop) 
            loss = criterion(outputs, targets)
            losses[index]=loss        

            sys.stdout.write('\r')
            sys.stdout.write('Evaluating Iter[%3d/%3d]'%(batch_idx+1, num_iter))
            sys.stdout.flush()

    losses = (losses-losses.min())/(losses.max()-losses.min())   
    input_loss = losses.reshape(-1,1).cpu()
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    idx_clean = torch.from_numpy(prob > 0.5)
    precision = torch.sum(~noise_idx[idx_clean]) / float(torch.sum(idx_clean))
    recall = torch.sum(~noise_idx[idx_clean]) / torch.sum(~noise_idx)
    print("GMM Precision: %2f Recall %2f"%(precision * 100., recall * 100.))
    return idx_clean

# Train
def train(epoch, dataloader):
    model.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    correct = 0
    total = 0

    for batch_idx, (inputs, targets,_, index) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs, return_sim=cfg.is_coop)

        if cfg.lnl_methods == "SCE":
            loss_per_sample = robust_criterion(outputs, targets)
            loss = loss_per_sample.mean()
        elif cfg.lnl_methods == "ELR":
            loss_per_sample = robust_criterion(index, outputs, targets) 
            loss = loss_per_sample.mean()
        elif cfg.lnl_methods == "GMM":
            loss_per_sample = criterion(outputs, targets) 
            if epoch <= cfg.warmup:
                loss = loss_per_sample.mean()
            else:
                loss = loss_per_sample[idx_clean[index]].mean()
        else:  # CE
            loss_per_sample = criterion(outputs, targets) 
            loss = loss_per_sample.mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).cpu().sum().item()

        sys.stdout.write('\r')
        sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d]\t total-loss: %.4f'
                %( epoch, cfg.epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

    return 100.*correct/total

# Test
def test(epoch, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs, return_sim=cfg.is_coop)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f\n" %(epoch, acc))
    return acc

# ======== Data ========
if cfg.dataset.startswith("cifar"):
    print("Loading CIFAR...")
    from dataloader import dataloader_cifar as dataloader
    loader = dataloader.cifar_dataloader(cfg.dataset, noise_mode=cfg.noise_mode, noise_ratio=cfg.noise_ratio,\
                                        batch_size=cfg.batch_size, num_workers=cfg.num_workers, root_dir=cfg.data_path, model=cfg.model)
    train_loader = loader.run('train')
    test_loader = loader.run('test')
    eval_loader = loader.run('eval')
elif cfg.dataset.startswith("cub"):
    print("Loading CUB-200-2011...")
    from dataloader import dataloader_cub as dataloader
    train_loader, eval_loader, test_loader = dataloader.build_loader(cfg)
elif cfg.dataset == "stanford_cars":
    print("Loading Stanford Cars...")
    from dataloader import dataloader_stanford_cars as dataloader
    train_loader, eval_loader, test_loader = dataloader.build_loader(cfg)
elif cfg.dataset == "tiny_imagenet":
    print("Loading Tiny-ImageNet...")
    from dataloader import dataloader_tiny_imagenet as dataloader
    train_loader, eval_loader, test_loader = dataloader.build_loader(cfg)

noise_idx = train_loader.dataset.noise_idx
num_class = cfg.num_class

# ======== Model ========
model, optimizer = load_clip(cfg)
model.cuda()
criterion = torch.nn.CrossEntropyLoss(reduction='none')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
if cfg.lnl_methods == "ELR":
    robust_criterion = ELRLoss(len(train_loader.dataset), num_class)
elif cfg.lnl_methods == "SCE":
    robust_criterion = SCELoss(alpha=1, beta=1, num_classes=num_class, reduction="none")

for epoch in range(1, cfg.epochs + 1):
    if cfg.lnl_methods == "GMM" and epoch > cfg.warmup:
        idx_clean = eval_train(eval_loader)
    train_acc = train(epoch, train_loader)
    test_acc = test(epoch, test_loader)
    scheduler.step()