import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch
import pandas as pd
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
        """ basic information """
        self.root = root
        self.resolution = resolution
        self.train = train
        self.num_class = 200
        os.makedirs(os.path.join(root, 'noise_file'), exist_ok=True)

        """ declare data augmentation """
        if self.train:
            self.transforms = transforms.Compose([
                        transforms.Resize((256, 256), Image.BILINEAR),
                        transforms.RandomCrop((resolution, resolution)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((resolution, resolution)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        self._load_metadata()

        data = []
        label = []
        for i in range(len(self.data)):
            data.append(os.path.join(self.root, "images", self.data.iloc[i].filepath))
            label.append(int(self.data.iloc[i].target - 1))

        if self.train:
            data_num = len(data)
            noise_file = os.path.join(self.root, 'noise_file', noise_mode + '_' + str(noise_ratio))

            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
            elif noise_mode == 'idn':
                images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                                        names=['img_id', 'filepath'])
                image_class_labels = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'),
                                                    sep=' ', names=['img_id', 'target'])
                train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                                sep=' ', names=['img_id', 'is_training_img'])

                data_idn = images.merge(image_class_labels, on='img_id')
                data_idn = data_idn.merge(train_test_split, on='img_id')
                data_tmp = data_idn[data_idn.is_training_img == 1]
                data_idn = []
                label_idn = []
                for i in range(len(data_tmp)):
                    data_idn.append(os.path.join(self.root, "images", data_tmp.iloc[i].filepath))
                    label_idn.append(int(data_tmp.iloc[i].target - 1))

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
                for i in range(data_num):
                    if i in noise_idx:
                        if noise_mode=='sym':
                            noiselabel = random.randint(0,199)
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

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

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
        
        return img, label
    

if __name__ == '__main__':
    test_set = ImageDataset(train=False, root="../data/cub-200-2011", resolution=224)
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=8, shuffle=True, batch_size=32)

    for i, (input, label) in enumerate(test_loader):
        pass
