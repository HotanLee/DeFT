from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch

            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[], log=''): 
        
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        os.makedirs(os.path.join(root_dir, 'noise_file'), exist_ok=True)
        noise_file = os.path.join(root_dir, 'noise_file', dataset + '_' + noise_mode + '_' + str(r))
     
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
                self.num_class = 10
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
                self.num_class = 100
            
            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
            elif noise_mode == 'idn':
                data = torch.from_numpy(train_data).float()
                targets = torch.from_numpy(np.array(train_label))
                dataset = zip(data, targets)
                noise_label = self.get_instance_noisy_label(self.r, dataset, targets, self.num_class)
                print("save noisy labels to %s ..."%noise_file)        
                json.dump(noise_label,open(noise_file,"w"))   
            elif noise_mode == 'real':
                if dataset == 'cifar10':
                    noise_label = torch.load('./data/cifar-10/CIFAR-10_human.pt')
                    noise_label = list(noise_label['worse_label'])
                else:
                    noise_label = torch.load('./data/cifar-100/CIFAR-100_human.pt')
                    noise_label = list(noise_label['noisy_label'])
                for i in range(len(noise_label)):
                    noise_label[i] = int(noise_label[i])
            else: # sym or asym or str
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r*50000)            
                noise_idx = idx[:num_noise]
                
                asym_transition = {}
                for i in range(self.num_class - 1):
                    asym_transition[i] = i + 1
                asym_transition[self.num_class - 1] = 0

                noise_class = list(range(self.num_class))
                random.shuffle(noise_class)
                noise_class = noise_class[:int(self.num_class / 2)]

                for i in range(50000):
                    if i in noise_idx:
                        if noise_mode=='sym':
                            noiselabel = random.randint(0, self.num_class - 1)
                            noise_label.append(noiselabel)
                        elif noise_mode=='asym':   
                            noiselabel = asym_transition[train_label[i]]
                            noise_label.append(noiselabel)  
                        elif noise_mode == 'str':
                            if train_label[i] in noise_class:
                                noiselabel = random.randint(0, self.num_class - 1)
                            else:
                                noiselabel = train_label[i]
                            noise_label.append(noiselabel)  
                    else:    
                        noise_label.append(train_label[i])   
                print("save noisy labels to %s ..."%noise_file)        
                json.dump(noise_label,open(noise_file,"w"))

            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            self.train_data = train_data
            self.noise_label = noise_label
            self.clean_label = train_label
            self.noise_idx = torch.zeros(50000, dtype=torch.bool)
            for i in range(50000):
                if train_label[i] != noise_label[i]:
                    self.noise_idx[i] = 1
            
            self.transition = np.zeros((self.num_class, self.num_class))
            num_per_class = np.zeros(self.num_class)
            for i in range(len(train_label)):
                num_per_class[train_label[i]] += 1
                self.transition[train_label[i]][noise_label[i]] += 1
            
            for i in range(self.num_class):
                self.transition[i] /= num_per_class[i]         
                
    def __getitem__(self, index):        
        if self.mode=='train':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            comple_target = np.random.choice(self.num_class)
            while comple_target == target:
                comple_target = np.random.choice(self.num_class)
            return img, target, comple_target, index  
        elif self.mode=='eval':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, target, index  
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_instance_noisy_label(self, n, dataset, labels, num_classes, feature_size=3*32*32, norm_std=0.1, seed=1): 
        # n -> noise_rate 
        # dataset -> mnist, cifar10 # not train_loader
        # labels -> labels (targets)
        # label_num -> class number
        # feature_size -> the size of input images (e.g. 28*28)
        # norm_std -> default 0.1
        # seed -> random_seed 
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
        
        
class cifar_dataloader():  
    def __init__(self, dataset, noise_ratio, noise_mode, batch_size, num_workers, root_dir, model, log='', noise_file=''):
        self.dataset = dataset
        self.r = noise_ratio
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.model = model
        self.log = log
        self.noise_file = noise_file
        if self.model == 'resnet' or self.model == 'vit':
            resolution = 32
        else:
            resolution = 224

        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.Resize(resolution),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                    ]) 
            self.transform_test = transforms.Compose([
                        transforms.Resize(resolution),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                    ])
        elif self.dataset=='cifar100':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize(resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
            self.transform_test = transforms.Compose([
                    transforms.Resize(resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
            
    def run(self,mode):
        if mode=='train':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="train",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='eval':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode="eval", noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)     
            return trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader          