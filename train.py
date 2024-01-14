# -*- coding: utf-8 -*-
#
from __future__ import division, print_function

import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import argparse
import numpy as np
import random
import datetime
import time

from network import AlexNetFc
from AllDeepbaselines_test import test_MAP, eval_val_dataset
from meters import AverageMeter, loss_store_init, print_loss, remark_loss, reset_loss
from dataset import UniRSDataset, RSTripletDataset

def cdist2(A, B):
    return  (A.pow(2).sum(1, keepdim = True)
             - 2 * torch.mm(A, B.t())
             + B.pow(2).sum(1, keepdim = True).t())

def calcTopKPostiveNegative_one(A_data, B_data, labels, topk=20):
    start1=time.time()
    A2B_dist = cdist2(A_data, B_data)
    end1=time.time()
    print('time1: ' + str(end1- start1), flush=True )
    #按照距离降序
    start2=time.time()
    _, A2B_topk = A2B_dist.topk(B_data.shape[0], dim=1, largest=False, sorted=True)
    end2=time.time()
    print('time2: ' + str(end2- start2), flush=True )
    
    #BxB
    tensor_labels = torch.tensor(labels, dtype=torch.float32).cuda()
    database_B_labels = tensor_labels[A2B_topk.view(-1), :].reshape(A_data.shape[0], B_data.shape[0], -1)
    query_A_labels = tensor_labels.unsqueeze(1)
    print((query_A_labels.shape, database_B_labels.shape), flush=True)
    all_imatch = torch.matmul(query_A_labels, database_B_labels.transpose(1, 2)).squeeze(1)
    A2B_positive_topk = torch.zeros((A_data.shape[0], topk), dtype=np.int)
    A2B_negative_topk = torch.zeros((A_data.shape[0], topk), dtype=np.int)
    print(('all_imatch', all_imatch.shape), flush=True)
    for idx in range(A_data.shape[0]):
        A2B_negative_topk[idx] = A2B_topk[idx, all_imatch[idx, :]==0][0:topk]
        A2B_positive_topk[idx] = A2B_topk[idx, all_imatch[idx, :]>0][-topk:]

    return A2B_positive_topk.cpu().numpy(), A2B_negative_topk.cpu().numpy()

def calcTopKPostiveNegative(A_data, B_data, labels, topk=20):
    start = time.time()
    A2B_positive_topk, A2B_negative_topk = calcTopKPostiveNegative_one(A_data, B_data, labels.copy(), topk)
    end = time.time()
    print('dot time: ' + str(end - start), flush=True)
    B2A_positive_topk, B2A_negative_topk = calcTopKPostiveNegative_one(B_data, A_data, labels.copy(), topk)
    print(('topK', topk, A2B_positive_topk.shape, A2B_negative_topk.shape, B2A_positive_topk.shape, B2A_negative_topk.shape), flush=True)
    return A2B_positive_topk, A2B_negative_topk, B2A_positive_topk, B2A_negative_topk

def train_hash(model_A, model_B, train_dataset, database_dataset_A, database_dataset_B,
        valid_dataset_A, valid_dataset_B, srcname_list, args):
    
    train_loader_A_B = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=args.workers)
    
    ## hashbit x class_count
    wInit = torch.nn.init.orthogonal_ (torch.empty (args.hash_bit, args.class_cnt))
    ## hashbit x class_count
    project_layer = torch.nn.Linear (args.hash_bit, args.class_cnt, bias = False).to(args.device)

    with torch.no_grad():
        print(project_layer.weight.shape)
        _ = project_layer.weight.copy_ (wInit.t())

    params_list = [{'params': model_A.parameters()},
                    {'params': model_B.parameters()},
                    {'params': project_layer.parameters()}]    
    optimizer = torch.optim.Adam(params_list, lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    
    loss_store = ["classification loss_A", "classification loss_B", 'hash_loss', 'A2B_triplet_loss', 'B2A_triplet_loss', 'triplet_loss', 'loss']
    loss_store = loss_store_init(loss_store)

    total_train_cnt = len(train_dataset.x_list)
    all_A_hashcodes = torch.randn(total_train_cnt, args.hash_bit, dtype=torch.float).to(args.device)
    all_B_hashcodes = torch.randn(total_train_cnt, args.hash_bit, dtype=torch.float).to(args.device)
    all_labels = np.array(train_dataset.y_list, dtype=np.int)
    print(all_labels.shape)
    topk=86 
    print(('topk', topk), flush=True)

    criterion = nn.CrossEntropyLoss().cuda()
    pdist = nn.PairwiseDistance(2)
    criterion_mse = nn.MSELoss().cuda()

    best_MAP = 0

    for epoch in range(0, args.max_epoch):
        lr_A = optimizer.param_groups[0]['lr']
        lr_B = optimizer.param_groups[1]['lr']
        print('epoch: {} lr_A: {:.8f} lr_B: {:.8f}'.format(epoch, lr_A, lr_B))

        all_A_hashcodes[all_A_hashcodes>=0] = 1
        all_A_hashcodes[all_A_hashcodes<0] = -1
        all_B_hashcodes[all_B_hashcodes>=0] = 1
        all_B_hashcodes[all_B_hashcodes<0] = -1

        A2B_positive_topk, A2B_negative_topk, B2A_positive_topk, B2A_negative_topk = calcTopKPostiveNegative(all_A_hashcodes, all_B_hashcodes, all_labels, topk=topk)
        train_dataset.updateTopK(A2B_positive_topk, A2B_negative_topk, B2A_positive_topk, B2A_negative_topk, topk=topk)
        
        project_layer.train()
        model_A.train()
        model_B.train()

        for batch_idx, (A_data, B2A_pos_data, B2A_neg_data, B_data, A2B_pos_data, A2B_neg_data, label_onehot, index) in enumerate(train_loader_A_B):
            optimizer.zero_grad()
            A_data = A_data.to(args.device)
            B2A_pos_data, B2A_neg_data = B2A_pos_data.to(args.device), B2A_neg_data.to(args.device)
            B_data = B_data.to(args.device)
            A2B_pos_data, A2B_neg_data = A2B_pos_data.to(args.device), A2B_neg_data.to(args.device)

            label = np.array([np.where(one==1)[0][0] for one in label_onehot], dtype=np.long)
            label = torch.from_numpy(label).to(args.device)

            A_anchor = model_A(A_data)
            A2B_pos, A2B_neg = model_B(A2B_pos_data), model_B(A2B_neg_data)
            B_anchor = model_B(B_data)
            B2A_pos, B2A_neg = model_A(B2A_pos_data), model_A(B2A_neg_data)

            y_pred_A = project_layer(A_anchor)
            y_pred_B = project_layer(B_anchor)
            
            classification_loss_A = criterion(y_pred_A, label)
            classification_loss_B = criterion(y_pred_B, label)
            
            hash_loss = torch.mean( (torch.abs(A_anchor)-1.0)**2) + torch.mean( (torch.abs(B_anchor)-1.0)**2) 
            # triplet loss
            A2B_triplet_loss = pdist(F.normalize(A_anchor, p=2, dim=1), F.normalize(A2B_pos, p=2, dim=1)) - pdist(F.normalize(A_anchor, p=2, dim=1), F.normalize(A2B_neg, p=2, dim=1))
            B2A_triplet_loss = pdist(F.normalize(B_anchor, p=2, dim=1), F.normalize(B2A_pos, p=2, dim=1)) - pdist(F.normalize(B_anchor, p=2, dim=1), F.normalize(B2A_neg, p=2, dim=1))
            A2B_triplet_loss, B2A_triplet_loss = F.relu(A2B_triplet_loss), F.relu(B2A_triplet_loss)
            A2B_triplet_loss, B2A_triplet_loss = A2B_triplet_loss.mean(), B2A_triplet_loss.mean()
            triplet_loss = A2B_triplet_loss + B2A_triplet_loss

            loss = classification_loss_A + classification_loss_B + hash_loss + args.lambda0 * triplet_loss

            loss.backward()
            optimizer.step()

            all_A_hashcodes[index, :] = A_anchor.clone().detach().data
            all_B_hashcodes[index, :] = B_anchor.clone().detach().data

            remark_loss(loss_store, classification_loss_A, classification_loss_B, hash_loss, A2B_triplet_loss, B2A_triplet_loss, triplet_loss, loss)
            print_loss(epoch, batch_idx, len(train_loader_A_B), srcname_list[0]+'_'+srcname_list[1], loss_store)

        print_loss(epoch, len(train_loader_A_B), len(train_loader_A_B), srcname_list[0]+'_'+srcname_list[1], loss_store)
        reset_loss(loss_store)

        scheduler.step()
        
        cur_avg_map = eval_val_dataset(epoch, model_A, model_B, database_dataset_A, database_dataset_B,
            valid_dataset_A, valid_dataset_B, srcname_list, args, best_MAP, eval_interval=10)
        if cur_avg_map > best_MAP:
            old_best = best_MAP
            best_MAP = cur_avg_map
            old_path = args.model_path+'_pl'
            if os.path.exists(old_path):
                os.remove(old_path)
            torch.save(project_layer, args.model_path+'_pl')

def main():
    
    seed = 13
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed+1)
    torch.manual_seed(seed+2)
    torch.cuda.manual_seed_all(seed+3)

    task_name='ours'
    parser = argparse.ArgumentParser( description='ours',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model
    parser.add_argument('--task_name', type=str, default='ours', help='task name')
    parser.add_argument('--model_type', type=str, default='alexnet', help='base model')
    # Hashing
    parser.add_argument('--hash_bit', type=int, default='32', help = 'hash bit')

    # Training 200
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--max_epoch', type=int, default=200, help='training epoch')
    parser.add_argument('--workers', type=int, default=8, help='number of data loader workers.')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--cm_centers_filepath', type=str, default='data/MRSSID/cm_kmeans_alexnet_centers256.npy', help='cm_centers_filepath')
    parser.add_argument('--data_name', type=str, default='MRSSID', help='MRSSID')
    parser.add_argument('--class_cnt', type=int, default='6', help='class count')
    parser.add_argument('--model_path', type=str, default='', help='')
    parser.add_argument('--lambda0', type=float, default=1, help='hyper-parameters 0')

    # Testing
    parser.add_argument('--Rstr', type=str, default='86', help='MAP@R')
    parser.add_argument('--T', type=float, default=0, help='Threshold for binary')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")

    task_name = args.task_name
    dir_tmp = 'data/' + str(args.data_name) + '/model/'+task_name+'/'
    if not os.path.exists(dir_tmp):
        os.makedirs(dir_tmp)
    args.model_path = os.path.join(dir_tmp, task_name+'_' + str(args.data_name) + '_' + args.model_type + '_' + str(args.hash_bit))
    
    args.Rlist = [int(one) for one in args.Rstr.split()]

    srcname_list=['MS', 'FC']
    prefix_list=['data/'+args.data_name+'/images/'+one+'_' for one in srcname_list]
    suffix_list=['.jpg', '.jpg']
    
    train_dataset_list, val_dataset_list, database_dataset_list = [], [], []
    for idx in range(len(srcname_list)):
        one_train_dataset = UniRSDataset('data/'+args.data_name+'/train.txt', prefix_list[idx], suffix_list[idx], is_train=True, batch_size=args.batch_size)
        one_val_dataset = UniRSDataset('data/'+args.data_name+'/test.txt', prefix_list[idx], suffix_list[idx], is_train=False, batch_size=args.batch_size)
        one_database_dataset = UniRSDataset('data/'+args.data_name+'/database.txt', prefix_list[idx], suffix_list[idx], is_train=False, batch_size=args.batch_size)
        train_dataset_list.append(one_train_dataset)
        val_dataset_list.append(one_val_dataset)
        database_dataset_list.append(one_database_dataset)

    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))

    cm_centers_filepath = args.cm_centers_filepath
    dict_kmeans_centers = np.load(cm_centers_filepath, allow_pickle=True).item()
    kmeans_centers_list = [dict_kmeans_centers['kmeans_'+one.lower()] for one in srcname_list]
    idx1,idx2=0,1
    model1 = AlexNetFc(kmeans_centers_list[idx2], args).to(args.device)
    model2 = AlexNetFc(kmeans_centers_list[idx1], args).to(args.device)
    one_train_dataset = RSTripletDataset('data/'+args.data_name+'/train.txt', [prefix_list[idx1], prefix_list[idx2]], [suffix_list[idx1], suffix_list[idx2]], is_train=True, batch_size=args.batch_size)
    train_hash(model1, model2, one_train_dataset, database_dataset_list[idx1], database_dataset_list[idx2],
            val_dataset_list[idx1], val_dataset_list[idx2], [srcname_list[idx1], srcname_list[idx2]], args)
            
if __name__ == '__main__':
    main()

