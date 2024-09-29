from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os

class imagenet_dataset(Dataset):
    def __init__(self, root_dir, transform, num_class):
        self.root = root_dir+'imagenet/val/'
        self.transform = transform
        self.val_data = []
        for c in range(num_class):
            imgs = os.listdir(self.root+str(c))
            for img in imgs:
                self.val_data.append([c,os.path.join(self.root,str(c),img)])                
                
    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')   
        img = self.transform(image) 
        return img, target
    
    def __len__(self):
        return len(self.val_data)

class webvision_dataset(Dataset): 
    def __init__(self, root_dir, transform, mode, num_class, pred=[], probability=[], log=''): 
        self.root = root_dir
        self.transform = transform
        self.mode = mode  
        self.num_class = num_class
     
        if self.mode=='test':
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img]=target                             
        else:    
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()    
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    train_imgs.append(img)
                    self.train_labels[img]=target            
            self.train_imgs = train_imgs         
                    
    def __getitem__(self, index):
        if self.mode=='train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(self.root+img_path).convert('RGB')   
            img = self.transform(image)
            comple_target = np.random.choice(self.num_class)
            while comple_target == target:
                comple_target = np.random.choice(self.num_class)
            return img, target, comple_target, index       
        elif self.mode=='eval':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(self.root+img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, index   
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)    

def build_loader(cfg):  
    transform_train = transforms.Compose([
            transforms.Resize((256, 256), Image.BILINEAR),
            transforms.RandomCrop((cfg.resolution, cfg.resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ]) 
    transform_test = transforms.Compose([
            transforms.Resize((256, 256), Image.BILINEAR),
            transforms.CenterCrop((cfg.resolution, cfg.resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])  
    transform_imagenet = transforms.Compose([
            transforms.Resize((256, 256), Image.BILINEAR),
            transforms.CenterCrop((cfg.resolution, cfg.resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])         

    train_dataset = webvision_dataset(root_dir=cfg.data_path, transform=transform_train, mode="train", num_class=cfg.num_class)                
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)     

    eval_dataset = webvision_dataset(root_dir=cfg.data_path, transform=transform_test, mode="eval", num_class=cfg.num_class)                
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)              
    
    test_dataset = webvision_dataset(root_dir=cfg.data_path, transform=transform_test, mode='test', num_class=cfg.num_class)      
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,pin_memory=True)               

    imagenet_val = imagenet_dataset(root_dir=cfg.data_path, transform=transform_imagenet, num_class=cfg.num_class)      
    imagenet_loader = DataLoader(dataset=imagenet_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, eval_loader, test_loader, imagenet_loader
