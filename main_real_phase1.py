import os
import sys
import random
import argparse
import numpy as np

import torch
from utils.config import _C as cfg
from utils.lnl_methods import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default="", help="path to config file")
parser.add_argument("--gpuid", default=None)

args = parser.parse_args()
cfg.defrost()
cfg.merge_from_file(args.cfg)
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

# Train
def train(epoch, dataloader):
    model.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    correct, total = 0, 0

    for batch_idx, (inputs, targets, comple_target, index) in enumerate(dataloader):
        inputs, targets, comple_target = inputs.cuda(), targets.cuda(), comple_target.cuda()
        pos_outputs, neg_outputs = model(inputs, return_neg=True)
        _, predicted = pos_outputs.max(1)

        logits_per_image = torch.cat([pos_outputs.detach().unsqueeze(-1), neg_outputs.unsqueeze(-1)], dim=-1)
        logits_per_image = F.softmax(logits_per_image, dim=-1)
        p_yes, p_no = logits_per_image[:, :, 0].squeeze(-1), logits_per_image[:, :, 1].squeeze(-1)
        _, neg_predicted = p_yes.max(1)
        is_max = neg_predicted == targets
        p_clean = p_yes[range(inputs.shape[0]), targets]
        idx_select = ((p_clean > 0.5) & is_max).cpu()

        if epoch == cfg.epochs:
            total_clean_idx[index] = idx_select

        p_no_logits = torch.log(torch.clamp(p_no, min=1e-5, max=1.))
        p_yes_logits = torch.log(torch.clamp(p_yes, min=1e-5, max=1.))
        if epoch <= cfg.warmup:
            loss_per_sample = robust_criterion(pos_outputs, targets)
            loss_cls = loss_per_sample.mean()
        else:
            loss_per_sample = criterion(pos_outputs, targets)
            loss_cls = loss_per_sample[idx_select].mean()
        
        if epoch <= 1:
            loss_neg = neg_criterion(p_no_logits, comple_target).mean() + neg_criterion(p_yes_logits, targets).mean()
        else: 
            pseudos = targets.clone()
            pseudos[~idx_select] = predicted[~idx_select]
            loss_neg = neg_criterion(p_no_logits, comple_target).mean() + neg_criterion(p_yes_logits, pseudos).mean()

        # total loss
        loss = loss_cls + loss_neg

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
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
    total_preds = []
    total_targets = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs, return_sim=True)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
            total_preds.append(predicted)
            total_targets.append(targets)

    acc = 100. * correct / total
    total_preds = torch.cat(total_preds)
    total_targets = torch.cat(total_targets)

    print("\n| Test Epoch #%d\t Accuracy: %.2f\n" %(epoch, acc))
    return acc

# ======== Data ========
if cfg.dataset == "clothing1m":
    from dataloader import dataloader_clothing1M as dataloader
    train_loader, _, test_loader = dataloader.build_loader(cfg)
elif cfg.dataset == "webvision":
    from dataloader import dataloader_webvision as dataloader
    train_loader, _, test_loader, imagenet_loader = dataloader.build_loader(cfg)
elif cfg.dataset.startswith("cifar"):
    from dataloader import dataloader_cifar as dataloader
    loader = dataloader.cifar_dataloader(cfg.dataset, noise_mode=cfg.noise_mode, noise_ratio=cfg.noise_ratio,\
                                        batch_size=cfg.batch_size, num_workers=cfg.num_workers, root_dir=cfg.data_path, model=cfg.model)
    train_loader = loader.run('train')
    test_loader = loader.run('test')

# ======== Model ========
model, optimizer = load_deft(cfg)
model.cuda()
criterion = torch.nn.CrossEntropyLoss(reduction='none')
neg_criterion = torch.nn.NLLLoss(reduction='none')
robust_criterion = SCELoss(alpha=1, beta=1, num_classes=cfg.num_class, reduction="none")
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
total_clean_idx = torch.zeros(len(train_loader.dataset), dtype=torch.bool)

for epoch in range(1, cfg.epochs + 1):
    train_acc = train(epoch, train_loader)
    test_acc = test(epoch, test_loader)
    if cfg.dataset == "webvision":
        imagenet_acc = test(epoch, imagenet_loader)

    if epoch == cfg.epochs:
        torch.save(total_clean_idx.numpy(), "./phase1/{}.pt".format(cfg.dataset))