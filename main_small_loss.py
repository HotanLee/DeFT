import os
import sys
import random
import argparse
import numpy as np

import torch
from utils.config import _C as cfg
from model import *

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default="", help="path to config file")
parser.add_argument("--noise_mode", default=None)
parser.add_argument("--noise_ratio", default=None)
parser.add_argument("--gpuid", default=None)

args = parser.parse_args()
cfg.defrost()
cfg.merge_from_file(args.cfg)
if args.noise_mode is not None:
    cfg.noise_mode = args.noise_mode
if args.noise_ratio is not None:
    cfg.noise_ratio = float(args.noise_ratio)
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

# Loss functions
def loss_coteaching(loss, y, t, forget_rate, ind, noise_or_not):
    ind_sorted =torch.argsort(loss.data)
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    precision = torch.sum(~noise_or_not[ind[ind_sorted[:num_remember].cpu()]])/float(num_remember)
    recall = torch.sum(~noise_or_not[ind[ind_sorted[:num_remember].cpu()]]) / torch.sum(~noise_or_not[ind])
    
    ind_update=ind_sorted[:num_remember]
    loss_update = loss[ind_update].mean()

    return loss_update, precision.data, recall.data


def train(epoch, train_loader):
    model.train()
    pure_ratio_list = []
    recall_list = []
    clean_loss = 0
    noisy_loss = 0
    num_iter = (len(train_loader.dataset) // train_loader.batch_size) + 1

    total = 0
    correct = 0 

    for batch_idx, (inputs, targets, _, index) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs1 = model(inputs)
        _, predicted1 = outputs1.max(1)
        total += targets.size(0)
        correct += predicted1.eq(targets).cpu().sum().item()

        loss = criterion(outputs1, targets)
        clean_step = loss[ ~ noise_idx[index]].mean().item()
        noisy_step = loss[noise_idx[index]].mean().item()
        clean_loss += clean_step
        noisy_loss += noisy_step

        loss, pure_ratio, recall = loss_coteaching(loss, outputs1, targets, rate_schedule[epoch - 1], index, noise_idx)
        pure_ratio_list.append(100*pure_ratio)
        recall_list.append(100*recall)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        sys.stdout.write('\r')
        sys.stdout.write('Epoch [%3d/%3d] Iter [%3d/%3d]\t Loss: %.4f clean-loss: %.4f noisy-loss: %.4f Precison: %.4f Recall: %.4f' 
                %(epoch, cfg.epochs, batch_idx + 1, num_iter, loss.item(), clean_loss / (batch_idx + 1), noisy_loss / (batch_idx + 1), \
                    np.sum(pure_ratio_list)/len(pure_ratio_list), np.sum(recall_list)/len(recall_list)))
        sys.stdout.flush()

    train_acc=float(correct)/float(total)
    
    return train_acc, pure_ratio_list, recall_list, noisy_loss / num_iter, clean_loss / num_iter

def test(epoch, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
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
elif cfg.dataset.startswith("cub"):
    print("Loading CUB-200-2011...")
    from dataloader import dataloader_cub as dataloader
    train_loader, _, test_loader = dataloader.build_loader(cfg)
elif cfg.dataset == "stanford_cars":
    print("Loading Stanford Cars...")
    from dataloader import dataloader_stanford_cars as dataloader
    train_loader, _, test_loader = dataloader.build_loader(cfg)
elif cfg.dataset == "tiny_imagenet":
    print("Loading Tiny-ImageNet...")
    from dataloader import dataloader_tiny_imagenet as dataloader
    train_loader, _, test_loader = dataloader.build_loader(cfg)

noise_idx = train_loader.dataset.noise_idx
num_class = cfg.num_class

if cfg.noise_mode == "str":
    rate_schedule = np.ones(cfg.epochs) * cfg.noise_ratio / 2
    rate_schedule[:cfg.warmup] = np.linspace(0, cfg.noise_ratio / 2, cfg.warmup)
else:
    rate_schedule = np.ones(cfg.epochs) * cfg.noise_ratio
    rate_schedule[:cfg.warmup] = np.linspace(0, cfg.noise_ratio, cfg.warmup)

# ======== Model ========
model, optimizer = load_clip(cfg)
model.cuda()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
criterion = torch.nn.CrossEntropyLoss(reduction='none')


for epoch in range(1, cfg.epochs + 1):
    train_acc, pure_ratio_list, recall_list, noisy_loss, clean_loss = train(epoch, train_loader)
    test_acc = test(epoch, test_loader)
    scheduler.step()
