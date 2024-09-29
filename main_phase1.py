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
parser.add_argument("--noise_mode", choices=['sym', 'idn'], default=None)
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

# Train
def train(epoch, dataloader):
    model.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    correct, total = 0, 0
    precision, recall = 0, 0

    for batch_idx, (inputs, targets, comple_target, index) in enumerate(dataloader):
        inputs, targets, comple_target = inputs.cuda(), targets.cuda(), comple_target.cuda()
        pos_outputs, neg_outputs = model(inputs, return_neg=True)
        _, predicted = pos_outputs.max(1)

        # computer clean probability
        logits_per_image = torch.cat([pos_outputs.detach().unsqueeze(-1), neg_outputs.unsqueeze(-1)], dim=-1)
        logits_per_image = F.softmax(logits_per_image, dim=-1)
        p_yes, p_no = logits_per_image[:, :, 0].squeeze(-1), logits_per_image[:, :, 1].squeeze(-1)
        p_clean = p_yes[range(inputs.shape[0]), targets]
        idx_select = (p_clean > 0.5).cpu()

        # save clean index
        if epoch == cfg.epochs:
            total_clean_idx[index] = idx_select

        # loss
        p_no_logits = torch.log(torch.clamp(p_no, min=1e-5, max=1.))
        p_yes_logits = torch.log(torch.clamp(p_yes, min=1e-5, max=1.))
        loss_per_sample = criterion(pos_outputs, targets)
        if epoch <= cfg.warmup:
            loss_cls = loss_per_sample.mean()
            loss_neg = neg_criterion(p_no_logits, comple_target).mean() + neg_criterion(p_yes_logits, targets).mean()
        else: 
            loss_cls = loss_per_sample[idx_select].mean()
            pseudos = targets.clone()
            pseudos[~idx_select] = predicted[~idx_select]
            loss_neg = neg_criterion(p_no_logits, comple_target).mean() + neg_criterion(p_yes_logits, pseudos).mean()
        loss = loss_neg + loss_cls

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total += targets.size(0)
        correct += predicted.eq(targets).cpu().sum().item()

        if torch.sum(idx_select) != 0:
            precision += torch.sum(~noise_idx[index[idx_select]]) / float(torch.sum(idx_select))
            recall += torch.sum(~noise_idx[index[idx_select]]) / torch.sum(~noise_idx[index])

        sys.stdout.write('\r')
        sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d]\t precision: %.4f recall: %.4f'
                %( epoch, cfg.epochs, batch_idx+1, num_iter, precision / (batch_idx + 1), recall / (batch_idx + 1)))
        sys.stdout.flush()

    return 100.*correct/total, precision / num_iter, recall / num_iter

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
if cfg.dataset.startswith("cifar"):
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
noise_label = torch.tensor(train_loader.dataset.noise_label).cuda()
clean_label = torch.tensor(train_loader.dataset.clean_label).cuda()
num_class = cfg.num_class


# ======== Model ========
model, optimizer = load_deft(cfg)
model.cuda()
criterion = torch.nn.CrossEntropyLoss(reduction='none')
neg_criterion = torch.nn.NLLLoss(reduction='none')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
total_clean_idx = torch.zeros(len(train_loader.dataset), dtype=torch.bool)  # save clean index in the last epoch

for epoch in range(1, cfg.epochs + 1):
    train_acc, precision, recall = train(epoch, train_loader)
    test_acc = test(epoch, test_loader)

    scheduler.step()
    if epoch == cfg.epochs:
        des = "./phase1/{}".format(cfg.dataset)
        if not os.path.exists(des):
            os.mkdir(des)
        torch.save(total_clean_idx.numpy(), os.path.join(des, "{}.pt".format(cfg.noise_mode + str(cfg.noise_ratio))))