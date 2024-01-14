# -*- coding: utf-8 -*-
#

import torchvision
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import os
import sys
import argparse
import numpy as np
import time

from metrics import mean_average_precision
from dataset import UniRSDataset, RSDataset
from network import AlexNetFc

def predict_hash_code(model, data_loader):  # data_loader is database_loader or test_loader
    model.eval()
    is_start = True
    for idx, (inputs, label, _) in enumerate(data_loader):
        inputs = Variable(inputs).cuda()
        #print(inputs.shape)
        #label = Variable(label).cuda()
        
        output = model(inputs)

        if is_start:
            all_output = output.data.cpu().float()
            all_label = label.float()
            is_start = False
        else:
            all_output = torch.cat((all_output, output.data.cpu().float()), 0)
            all_label = torch.cat((all_label, label.float()), 0)

    return all_output.cpu().numpy(), all_label.cpu().numpy().astype(np.int8)

def eval_val_dataset(epoch, model_sar, model_h, database_dataset_sar, database_dataset_h,
        valid_dataset_sar, valid_dataset_h, srcname_list, args, best_MAP, eval_interval=30, isSavemodel=True):
    if True:
        if epoch%eval_interval == 0 or epoch == (args.max_epoch-1):
            print('eval, epoch: %d' % epoch)
            mapval_list=list()
            model_list=[model_sar, model_h]
            database_dataset_list, val_dataset_list = [database_dataset_sar, database_dataset_h], [valid_dataset_sar, valid_dataset_h]
            for src1_idx in range(len(srcname_list)):
                val_loader = DataLoader(val_dataset_list[src1_idx], batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
                val_model = model_list[src1_idx]
                for src2_idx in range(len(srcname_list)):
                    if src1_idx == src2_idx:
                        continue
                    database_loader = DataLoader(database_dataset_list[src2_idx], batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
                    database_model = model_list[src2_idx]
                    task_name=srcname_list[src1_idx]+'2'+srcname_list[src2_idx]
                    print('======eval '+task_name+'======', flush=True)
                    curMap, _, _, _, _ = test_MAP(val_model, database_model, val_loader, database_loader, args)
                    mapval_list.append((task_name, curMap))
                    print((task_name, curMap), flush=True)
            cur_avg_map = sum([one[1] for one in mapval_list])/len(mapval_list)
            if cur_avg_map > best_MAP:
                old_best = best_MAP
                best_MAP = cur_avg_map
                task_name=srcname_list[0]+'2'+srcname_list[1]
                for idx, srcname in enumerate(srcname_list):
                    old_path = args.model_path+'_'+srcname
                    if isSavemodel:
                        if os.path.exists(old_path):
                            os.remove(old_path)
                        torch.save(model_list[idx], args.model_path+'_'+srcname)

            for task_name, curMap in mapval_list:
                print((task_name, curMap), flush=True)
            print('cur_map, best_map: {:.4f}, {:.4f}'.format(cur_avg_map, best_MAP), flush=True)
    return best_MAP

def generate_hashcodes_and_similarity_matrix(test_hash, database_hash, test_labels, database_labels, output_filename, output_hash_filename, args):

    print('start to generate_hashcodes_and_similarity_matrix', flush=True)
    
    database_hash[database_hash>=args.T] = 1
    database_hash[database_hash<args.T] = -1

    test_hash[test_hash>=args.T] = 1
    test_hash[test_hash<args.T] = -1
    
    sim = np.dot(database_hash, test_hash.T)
    
    np_y_test = np.array(test_labels).astype(np.int8)
    np_y_database = np.array(database_labels).astype(np.int8)
    
    np.save(output_filename, { 'sim':sim, 'y_test':np_y_test, 'y_database':np_y_database } )

    y_test = np.array([np.where(one==1)[0][0] for one in np_y_test], dtype=np.int16)
    y_database = np.array([np.where(one==1)[0][0] for one in np_y_database], dtype=np.int16)

    np.save(output_hash_filename, {'hash_test': np.array(test_hash).astype(np.int8), 'y_test': y_test, 
                'hash_database': np.array(database_hash).astype(np.int8), 'y_database': y_database})
    
    MAP, R, APx = mean_average_precision(database_hash, test_hash, database_labels, test_labels, args.Rlist[0], args.T)
    print('R={}, MAP {:.4f}, Recall {:.4f}'.format(args.Rlist[0], MAP, R), flush=True)

def test_MAP(model_test, model_database, test_loader, database_loader, args):
    print('start to model database', flush=True)
    start = time.time()
    database_hash, database_labels = predict_hash_code(model_database, database_loader)
    end = time.time()
    print('predict database time:'+str(end-start), flush=True)
    print(database_hash[0])
    print(database_labels[0])
    print(database_hash.shape)
    print(database_labels.shape)
    print('start to testset', flush=True)
    start = end
    test_hash, test_labels = predict_hash_code(model_test, test_loader)
    end = time.time()
    print('predict test time:'+str(end-start), flush=True)
    print(test_hash[0])
    print(test_labels[0])
    print(test_hash.shape)
    print(test_labels.shape)
    print('Calculate MAP.....', flush=True)
    start = end

    argsR_list = args.Rlist
    MAP_list = []
    R_list = []
    APx_list = []
    str_MAP='eval_MAP:\t'
    str_R='R:\t'
    str_APx=['APx:\t' for i in range(len(test_hash))]
    for _, argsR in enumerate(argsR_list):
        MAP, R, APx = mean_average_precision(database_hash.copy(), test_hash.copy(), database_labels.copy(), test_labels.copy(), argsR, args.T)
        MAP_list.append(MAP)
        R_list.append(R)
        APx_list.append(APx)
        str_MAP += '{:.4f}'.format(MAP) + '\t'
        str_R += str(R) + '\t'
        for i, val in enumerate(APx):
            str_APx[i] += str(val) + '\t'
    
    print(str_MAP)
    print(str_R)
    
    MAP, R, APx = MAP_list[0], R_list[0], APx_list[0]
    end = time.time()
    print('MAP time:'+str(end-start), flush=True)
    print('R={}, MAP {:.4f}, Recall {:.4f}'.format(argsR_list[0], MAP, R), flush=True)

    return MAP, test_hash, database_hash, test_labels, database_labels

def main():    
    parser = argparse.ArgumentParser( description='AllDeepMethods_test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model
    parser.add_argument('--task_name', type=str, default='dsmhn', help='null_task')
    parser.add_argument('--model_type', type=str, default='alexnet', help='base model')
    # Hashing
    parser.add_argument('--hash_bit', type=int, default=32, help = 'hash bit')

    # Test
    parser.add_argument('--workers', type=int, default=4, help='number of data loader workers.')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--data_name', type=str, default='MRSSID', help='MRSSID')
    
    parser.add_argument('--Rstr', type=str, default='86', help='MAP@R')
    parser.add_argument('--T', type=float, default=0, help='Threshold for binary')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")

    args.Rlist = [int(one) for one in args.Rstr.split()]

    task_name = args.task_name
    dir_tmp = 'data/' + str(args.data_name) + '/model/'+task_name+'/'
    args.model_path = os.path.join(dir_tmp, task_name+'_' + str(args.data_name) + '_' + args.model_type + '_' + str(args.hash_bit))
    results_output_dir = 'data/'+args.data_name+'/results/'+task_name+'/'
    if not os.path.exists(results_output_dir):
        os.makedirs(results_output_dir)

    srcname_list=['MS', 'FC']
    prefix_list=['data/'+args.data_name+'/images/'+one+'_' for one in srcname_list]
    suffix_list=['.jpg', '.jpg']
    
    test_dataset_list, database_dataset_list = [], []
    for idx in range(len(srcname_list)):
        one_test_dataset = UniRSDataset('data/'+args.data_name+'/test.txt', prefix_list[idx], suffix_list[idx], is_train=False, batch_size=args.batch_size)
        one_database_dataset = UniRSDataset('data/'+args.data_name+'/database.txt', prefix_list[idx], suffix_list[idx], is_train=False, batch_size=args.batch_size)
        test_dataset_list.append(one_test_dataset)
        database_dataset_list.append(one_database_dataset)

    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))
    
    mapval_list=list()
    srcidx_list = [0,1]
    for src1_idx in srcidx_list:
        test_loader = DataLoader(test_dataset_list[src1_idx], batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_model=torch.load(args.model_path+'_'+srcname_list[src1_idx]).to(args.device)
        for src2_idx in srcidx_list:
            if src1_idx == src2_idx:
                continue
            database_loader = DataLoader(database_dataset_list[src2_idx], batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
            database_model=torch.load(args.model_path+'_'+srcname_list[src2_idx]).to(args.device)
            retrieval_task_name=srcname_list[src1_idx]+'2'+srcname_list[src2_idx]
            print('======eval '+retrieval_task_name+'======', flush=True)
            curMap, test_hash, database_hash, test_labels, database_labels = test_MAP(test_model, database_model, test_loader, database_loader, args)
            mapval_list.append((retrieval_task_name, curMap))
            print((retrieval_task_name, curMap), flush=True)
            output_hashcode_src1tosrc2_path=os.path.join(results_output_dir, task_name+'_'+args.model_type+'_hashcodes_'+srcname_list[src1_idx]+'2'+srcname_list[src2_idx]+'_h'+str(args.hash_bit))
            sim_src1tosrc2_path=os.path.join(results_output_dir, task_name+'_'+args.model_type+'_sim_'+srcname_list[src1_idx]+'2'+srcname_list[src2_idx]+'_h'+str(args.hash_bit))
            generate_hashcodes_and_similarity_matrix(test_hash, database_hash, test_labels, database_labels, sim_src1tosrc2_path, output_hashcode_src1tosrc2_path, args)

            database_model = database_model.to('cpu')
            del database_model
            torch.cuda.empty_cache()
        test_model = test_model.to('cpu')
        del test_model
        torch.cuda.empty_cache()

    cur_avg_map = sum([one[1] for one in mapval_list])/len(mapval_list)
    mapval_list.append(('All', cur_avg_map))

    for retrieval_task_name, curMap in mapval_list:
        print((retrieval_task_name, curMap), flush=True)

if __name__ == '__main__':
    main()

