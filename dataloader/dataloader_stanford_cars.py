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
            elif noise_mode == 'idn':
                data_idn = []
                label_idn = []
                des_root = os.path.join(self.root, 'train')
                folders = os.listdir(des_root)
                folders.sort()
                for class_id, folder in enumerate(folders):
                    files = os.listdir(os.path.join(des_root, folder))
                    for file in files:
                        data_path = os.path.join(des_root, folder, file)
                        data_idn.append(data_path)
                        label_idn.append(class_id)

                print("image loaded")
                resized_img = []
                for img_path in data_idn:
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (64, 64))
                    img = img[:, :, ::-1] # BGR to RGB.
                    img = np.expand_dims(np.array(Image.fromarray(img)), 0)
                    resized_img.append(img)

                resized_img = np.concatenate(resized_img)
                print("shape of data: ", resized_img.shape)
                data_idn = torch.from_numpy(resized_img).float()
                targets = torch.from_numpy(np.array(label_idn))
                dataset = zip(data_idn, targets)
                print("generating noisy labels...")
                print("Noise_ratio: ", noise_ratio)
                noise_label = self.get_instance_noisy_label(noise_ratio, dataset, targets, self.num_class)
                print("save noisy labels to %s ..."%noise_file)
                json.dump(noise_label,open(noise_file,"w"))
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
                        else:
                            pass           
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

    def get_instance_noisy_label(self, n, dataset, labels, num_classes, feature_size=3*64*64, norm_std=0.1, seed=1): 
        from math import inf
        from scipy import stats

        print("building dataset...")
        label_num = num_classes
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed(int(seed))

        P = []
        flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
        flip_rate = flip_distribution.rvs(labels.shape[0])

        if isinstance(labels, list):
            labels = torch.FloatTensor(labels)
        labels = labels.cuda()

        W = np.random.randn(label_num, feature_size, label_num)


        W = torch.FloatTensor(W).cuda()
        for i, (x, y) in enumerate(dataset):
            # 1*m *  m*10 = 1*10
            x = x.cuda()
            A = x.view(1, -1).mm(W[y]).squeeze(0)
            A[y] = -inf
            A = flip_rate[i] * torch.nn.functional.softmax(A, dim=0)
            A[y] += 1 - flip_rate[i]
            P.append(A)
        P = torch.stack(P, 0).cpu().numpy()
        l = [i for i in range(label_num)]
        new_label = [int(np.random.choice(l, p=P[i])) for i in range(labels.shape[0])]
        record = [[0 for _ in range(label_num)] for i in range(label_num)]

        for a, b in zip(labels, new_label):
            a, b = int(a), int(b)
            record[a][b] += 1


        pidx = np.random.choice(range(P.shape[0]), 1000)
        cnt = 0
        for i in range(1000):
            if labels[pidx[i]] == 0:
                a = P[pidx[i], :]
                cnt += 1
            if cnt >= 10:
                break

        return new_label 

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
