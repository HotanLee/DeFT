import os
import sys
import random
import argparse
import numpy as np

import torch
from utils.config import _C as cfg
from model import *
import timm

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default="", help="path to config file")
parser.add_argument("--noise_mode", default=None)
parser.add_argument("--noise_ratio", default=None)
parser.add_argument("--gpuid", default=None)
parser.add_argument("--backbone", default=None)

args = parser.parse_args()
cfg.defrost()
cfg.merge_from_file(args.cfg)
if args.noise_mode is not None:
    cfg.noise_mode = args.noise_mode
if args.noise_ratio is not None:
    cfg.noise_ratio = float(args.noise_ratio)
if args.gpuid is not None:
    cfg.gpuid = int(args.gpuid)
if args.backbone is not None:
    cfg.backbone = args.backbone

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

    for batch_idx, (inputs, targets, _, index) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        loss_per_sample = criterion(outputs, targets)
        loss = loss_per_sample[total_clean_idx[index]].mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total += targets.size(0)
        correct += predicted.eq(targets).cpu().sum().item()

        sys.stdout.write('\r')
        sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.3f'
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
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
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
    from dataloader import dataloader_cub as dataloader
    train_loader, _, test_loader = dataloader.build_loader(cfg)
elif cfg.dataset == "stanford_cars":
    from dataloader import dataloader_stanford_cars as dataloader
    train_loader, _, test_loader = dataloader.build_loader(cfg)
elif cfg.dataset == "tiny_imagenet":
    from dataloader import dataloader_tiny_imagenet as dataloader
    train_loader, _, test_loader = dataloader.build_loader(cfg)
num_class = cfg.num_class


# ======== Model ========
if cfg.backbone == 'vit':
    model = timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=True, num_classes=num_class, pretrained_cfg_overlay=dict(file='./model/weights/vit.npz'))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-5)
elif cfg.backbone == 'resnet':
    model = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=num_class, pretrained_cfg_overlay=dict(file='./model/weights/resnet.pth'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
elif cfg.backbone == 'convnext':
    model = timm.create_model('convnext_tiny.fb_in1k', pretrained=True, num_classes=num_class, pretrained_cfg_overlay=dict(file='./model/weights/convnext.pth'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
elif cfg.backbone == 'mae':
    model = timm.create_model('vit_base_patch16_224.mae', pretrained=True, num_classes=num_class, pretrained_cfg_overlay=dict(file='./model/weights/mae.pth'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
else:
    model, optimizer = load_clip(cfg)
    cfg.backbone == 'clip'
model.cuda()
criterion = torch.nn.CrossEntropyLoss(reduction='none')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
total_clean_idx = torch.load("./phase1/{}/{}.pt".format(cfg.dataset, cfg.noise_mode + str(cfg.noise_ratio)))
best_acc = 0

for epoch in range(1, cfg.epochs + 1):
    train_acc = train(epoch, train_loader)
    test_acc = test(epoch, test_loader)
    best_acc = max(best_acc, test_acc)
    if epoch == cfg.epochs:
        print("Best Acc: %.2f Last Acc: %.2f" % (best_acc, test_acc))

    scheduler.step()