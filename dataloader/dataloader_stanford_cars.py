import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch
import json
import random


def build_loader(cfg):

    train_set = ImageDataset(train=True, root=cfg.data_path, resolution=cfg.resolution, noise_mode=cfg.noise_mode, noise_ratio=cfg.noise_ratio)
    train_loader = torch.utils.data.DataLoader(train_set, num_workers=cfg.num_workers, shuffle=True, batch_size=cfg.batch_size)

    eval_set = ImageDataset(train=True, root=cfg.data_path, resolution=cfg.resolution, noise_mode=cfg.noise_mode, noise_ratio=cfg.noise_ratio)
    eval_loader = torch.utils.data.DataLoader(eval_set, num_workers=cfg.num_workers, shuffle=False, batch_size=cfg.batch_size)

    test_set = ImageDataset(train=False, root=cfg.data_path, resolution=cfg.resolution, noise_mode=cfg.noise_mode, noise_ratio=cfg.noise_ratio)
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=cfg.num_workers, shuffle=True, batch_size=cfg.batch_size)

    return train_loader, eval_loader, test_loader


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 train: bool,
                 root: str,
                 resolution: int,
                 noise_mode='sym',
                 noise_ratio=0.0,
                 ):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.root = root
        self.resolution = resolution
        self.train = train
        os.makedirs(os.path.join(root, 'noise_file'), exist_ok=True)

        """ declare data augmentation """
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        if self.train:
            self.transforms = transforms.Compose([
                        transforms.Resize((256, 256), Image.BILINEAR),
                        transforms.RandomCrop((resolution, resolution)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((resolution, resolution)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                ])

        """ read all data information """
        if self.train:
            data, label = self.getDataInfo(os.path.join(root, "train"))
        else:
            data, label = self.getDataInfo(os.path.join(root, "test"))

        if self.train:
            data_num = len(data)
            noise_file = os.path.join(self.root, 'noise_file', noise_mode + '_' + str(noise_ratio))

            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
            else:
                noise_label = []
                idx = list(range(data_num))
                random.shuffle(idx)
                num_noise = int(noise_ratio * data_num)            
                noise_idx = idx[:num_noise]
                asym_transition = {}
                for i in range(self.num_class - 1):
                    asym_transition[i] = i + 1
                asym_transition[self.num_class - 1] = 0
                noise_class = list(range(self.num_class))
                random.shuffle(noise_class)
                noise_class = noise_class[:int(self.num_class / 2)]

                for i in range(data_num):
                    if i in noise_idx:
                        if noise_mode=='sym':
                            noiselabel = random.randint(0, self.num_class - 1)
                            noise_label.append(noiselabel)
                        elif noise_mode=='asym':   
                            noiselabel = asym_transition[label[i]]
                            noise_label.append(noiselabel) 
                        elif noise_mode == 'str':
                            if label[i] in noise_class:
                                # noiselabel = random.randint(0, self.num_class - 1)
                                noiselabel = asym_transition[label[i]]
                            else:
                                noiselabel = label[i]
                            noise_label.append(noiselabel)                  
                    else:    
                        noise_label.append(label[i])   
                print("save noisy labels to %s ..."%noise_file)        
                json.dump(noise_label, open(noise_file,"w"))  

            self.data = data
            self.label = noise_label
            self.noise_idx = torch.zeros(data_num, dtype=torch.bool)
            for i in range(data_num):
                if label[i] != noise_label[i]:
                    self.noise_idx[i] = 1
            self.noise_label = noise_label
            self.clean_label = label
        else:
            self.data = data
            self.label = label

    def getDataInfo(self, root):
        data = []
        label = []
        folders = os.listdir(root)
        folders.sort() # sort by alphabet
        self.num_class = len(folders)
        for class_id, folder in enumerate(folders):
            files = os.listdir(os.path.join(root, folder))
            for file in files:
                data_path = os.path.join(root, folder, file)
                data.append(data_path)
                label.append(class_id)
        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get data information
        img = cv2.imread(self.data[index])
        img = img[:, :, ::-1] # BGR to RGB.
        img = Image.fromarray(img)
        img = self.transforms(img)
        
        label = self.label[index]
        
        if self.train:
            comple_target = np.random.choice(self.num_class)
            while comple_target == label:
                comple_target = np.random.choice(self.num_class)
            return img, label, comple_target, index
        
        # return img, sub_imgs, label, sub_boundarys
        return img, label