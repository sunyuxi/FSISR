# -*- coding: utf-8 -*-
#

import numpy as np
import torch
from torch.utils.data import Dataset
import sys
import random
import rasterio
import os
import cv2
from PIL import Image
import torchvision

################################
# dataset load
################################

def load_dataset(file_path, is_train=False, batch_size=None):
    x_list = []
    y_list = []

    all_line_list = [one.strip() for one in open(file_path)]
    if is_train:
        assert batch_size != None
        all_line_cnt = len(all_line_list)
        if all_line_cnt % batch_size != 0:
            #random.shuffle(all_line_list)
            padding_cnt = (int(all_line_cnt/batch_size)+1)*batch_size-all_line_cnt
            assert (all_line_cnt+padding_cnt)%batch_size == 0
            print(('padding_cnt', padding_cnt, all_line_cnt, batch_size))
            all_line_list = all_line_list + all_line_list[0:padding_cnt]
    print(('all_line_list', len(all_line_list)), flush=True)
    cnt = 0
    for one in all_line_list:
        arr = one.strip().split()
        x = arr[0]
        y = [ int(i) for i in arr[1:] ]
        x_list.append(x)
        y_list.append(y)

        cnt += 1
        if cnt%2000==0:
            print('load_dataset:' + str(cnt), flush=True)
            #pass
            #break

    return x_list, np.array(y_list)

def load_singleimg(img_name, prefix, buffix, is_train=True):
    if is_train:
        data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    else:
        data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
    
    input_filepath=prefix + img_name + buffix
    img_data = Image.open(input_filepath).convert('RGB')
    x = data_transforms(img_data)
    
    return x

class RSTripletDataset(Dataset):

    def __init__(self, img_filepath, prefix_list=[], suffix_list=[], is_train=False, batch_size=None):
        self.is_train = is_train
        self.img_filepath=img_filepath
        
        self.x_list, self.y_list = load_dataset(self.img_filepath, is_train, batch_size)
        print(self.y_list.shape)
        self.prefix_list, self.suffix_list = prefix_list, suffix_list
        assert len(self.prefix_list)==len(self.suffix_list)
    
    def updateTopK(self, A2B_positive_topk, A2B_negative_topk, B2A_positive_topk, B2A_negative_topk, topk=20):
        total_train_cnt = len(self.x_list)
        self.A2B_positive_idx = A2B_positive_topk[ np.arange(total_train_cnt), np.random.choice(topk, size=total_train_cnt) ]
        self.A2B_negative_idx = A2B_negative_topk[ np.arange(total_train_cnt), np.random.choice(topk, size=total_train_cnt) ]
        self.B2A_positive_idx = B2A_positive_topk[ np.arange(total_train_cnt), np.random.choice(topk, size=total_train_cnt) ]
        self.B2A_negative_idx = B2A_negative_topk[ np.arange(total_train_cnt), np.random.choice(topk, size=total_train_cnt) ]

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        img_list=list()
        positive_negative_list = [[self.B2A_positive_idx[idx], self.B2A_negative_idx[idx]], [self.A2B_positive_idx[idx], self.A2B_negative_idx[idx]]]
        for imgtype_idx in range(len(self.prefix_list)):
            prefix, suffix = self.prefix_list[imgtype_idx], self.suffix_list[imgtype_idx]
            img = load_singleimg(self.x_list[idx], prefix, suffix, is_train=self.is_train)
            img_list.append(img)
            positive_negative = positive_negative_list[imgtype_idx]
            img_p = load_singleimg(self.x_list[positive_negative[0]], prefix, suffix, is_train=self.is_train)
            img_n = load_singleimg(self.x_list[positive_negative[1]], prefix, suffix, is_train=self.is_train)
            img_list.append(img_p)
            img_list.append(img_n)

        return *img_list, self.y_list[idx], torch.from_numpy(np.array(idx))

class RSDataset(Dataset):

    def __init__(self, img_filepath, prefix_list=[], suffix_list=[], is_train=False, batch_size=None):
        self.is_train = is_train
        self.img_filepath=img_filepath
        
        self.x_list, self.y_list = load_dataset(self.img_filepath, is_train, batch_size)
        print(self.y_list.shape)
        self.prefix_list, self.suffix_list = prefix_list, suffix_list
        assert len(self.prefix_list)==len(self.suffix_list)

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        img_list=list()
        for imgtype_idx in range(len(self.prefix_list)):
            prefix, suffix = self.prefix_list[imgtype_idx], self.suffix_list[imgtype_idx]
            img = load_singleimg(self.x_list[idx], prefix, suffix, is_train=self.is_train)
            img_list.append(img)

        return *img_list, self.y_list[idx], torch.from_numpy(np.array(idx))

class UniRSDataset(Dataset):

    def __init__(self, img_filepath, prefix='', suffix='', is_train=False, batch_size=None):
        self.is_train = is_train
        self.img_filepath=img_filepath
        
        self.x_list, self.y_list = load_dataset(self.img_filepath, is_train, batch_size)
        print(self.y_list.shape)
        self.prefix, self.suffix = prefix, suffix

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        img = load_singleimg(self.x_list[idx], self.prefix, self.suffix, is_train=self.is_train)
        
        return img, self.y_list[idx], torch.from_numpy(np.array(idx))

class PairRSDataset(Dataset):

    def __init__(self, img_filepath, prefix_list=[], suffix_list=[], is_train=False, batch_size=None):
        self.is_train = is_train
        self.img_filepath=img_filepath
        
        self.x_list, self.y_list = load_dataset(self.img_filepath, is_train, batch_size)
        print(self.y_list.shape)
        self.prefix_list, self.suffix_list = prefix_list, suffix_list
        assert len(self.prefix_list)==len(self.suffix_list)
        
        if is_train:
            dict_class2list = {}
            for idx, one in enumerate(self.x_list):
                arr_tmp = one.split('_')
                if arr_tmp[-1] not in dict_class2list:
                    dict_class2list[arr_tmp[-1]] = list()
                dict_class2list[arr_tmp[-1]].append(idx)

            self.dict_class2list = dict_class2list
            
            self.classlabel_list = sorted([one for one in dict_class2list.keys()])
            print(('classlabel_list', self.classlabel_list))
            
    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):

        if self.is_train:
            positive_label = self.x_list[idx].split('_')[-1]
            neg_label = random.choice([one for one in self.classlabel_list if one != positive_label])
            
            neg_idx=random.choice(self.dict_class2list[neg_label])
            pos_idx=random.choice(self.dict_class2list[positive_label])
            # anchor, positive, negative
            dataidx_list=[idx, pos_idx, neg_idx, neg_idx]
            img_list=list()
            for imgtype_idx in range(len(self.prefix_list)):
                prefix, suffix = self.prefix_list[imgtype_idx], self.suffix_list[imgtype_idx]
                img = load_singleimg(self.x_list[dataidx_list[imgtype_idx]], prefix, suffix, is_train=self.is_train)
                img_list.append(img)

            return *img_list, self.y_list[idx], torch.from_numpy(np.array(idx))
        else:
            print('Not Implemented')
            assert False