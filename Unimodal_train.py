# -*- coding: utf-8 -*-
import argparse
import copy
import csv
from datetime import datetime
import math
import shutil
import sys
import json
import time
import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from models.MaskModel import VideoClassifier, AudioClassifier
from MyHelper import get_VA_data_loaders
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,average_precision_score
from torch.utils.tensorboard import SummaryWriter
import argparse

import random

sys.path.append('..')
sys.path.append('../..')
def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
def parse_options():
    parser = argparse.ArgumentParser(description='Img-Text Dataset Config')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='whether to use gpu')
    parser.add_argument('--random_seed', type=int, default=42,help='random seed')
    parser.add_argument('--data_path', type=str, default='/data/php/kinetics_sound')
    parser.add_argument('--max-seq-len', type=int, default=128)
    parser.add_argument('--pre-train', type=bool, default=False)
    parser.add_argument('--only_test', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=70,
                        help='the total number of epoch in the training')
    parser.add_argument('--num_workers', type=int , default=0)
    parser.add_argument('--modal', type=str, default='audio',choices=['audio','video'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr_decay_epoch', type=int, default=50,
                        help='the number of epoch to decay lr')
    parser.add_argument('--fusion_method', type=str, default='concat')
    parser.add_argument('--conv1_sparsity', type=float, default=0.5,help='conv1 sparsity')
    parser.add_argument('--sparsity', type=float, default=0.5, help='sparsity')
    parser.add_argument('--dataset',default='KS',type=str,help='dataset name')
    parser.add_argument('--batch-size', default=64, type=int,help='train batchsize')
    parser.add_argument('--test-batch-size', default=16, type=int,help='train batchsize')
    parser.add_argument('--n_classes', type=int, default=31)
    parser.add_argument("--our_model", default="SEBM",type=str,
                    help="Our Model's Name")
    parser.add_argument("--test-only",dest="test_only",
                        help="Only test the model",action="store_true",)
    parser.add_argument("--device", default="cuda", type=str,
                        help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument('--gpu_id', default=0, type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    config = parser.parse_args()
    return config

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)    
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(config, model, criterion, optimizer, train_loader,scheduler,epoch):

    model.train()
    all_label, all_out, all_loss, all_score, all_label_single = [], [], [], [], []
    train_tqdm = tqdm(train_loader)
    for index,batch in enumerate(train_tqdm):
        optimizer.zero_grad()
        spectrogram, images, label = batch

        if config.modal=='video':
            _,out = model(images.float().cuda(),2)
            net_type = 'video_net'
            
        elif config.modal=='audio' :
            _,out = model(spectrogram.unsqueeze(1).float().cuda(),2)
            net_type = 'audio_net'
        else:
            raise NotImplementedError('Incorrect modal name {}'.format(config.modal))
        
        loss = criterion(out, label.cuda())
        label_single = torch.argmax(label,1)
        loss.backward()              
        optimizer.step()
        temp_out = out.detach().cpu()
        preds = torch.argmax(temp_out, 1)
        all_score += preds.numpy().tolist()
        all_out += out.detach().cpu().numpy().tolist()
        all_label += label
        all_loss.append(loss.item())
        train_tqdm.set_description('Loss: %f' % (np.mean(all_loss)))
        all_label_single +=label_single.detach().cpu().tolist()
    scheduler.step()
    acc = accuracy_score(all_label_single, all_score)*100
    return np.mean(all_loss), acc

def evaluate(config, model, criterion, valid_loader,epoch):
    model.eval()
    all_label, all_out, all_loss, all_score, all_label_single = [], [], [], [], []
    softmax = nn.Softmax(dim=1)
    for batch in tqdm(valid_loader):
        with torch.no_grad():
            spectrogram, images, label = batch
            if config.modal=='video':
                _,out = model(images.float().cuda(),2)
            elif config.modal=='audio' :
                _,out = model(spectrogram.unsqueeze(1).float().cuda(),2)
            else:
                raise NotImplementedError('Incorrect modal name {}'.format(config.modal))
            loss = criterion(out, label.cuda())  

            label_single = torch.argmax(label,1)
            all_loss.append(loss.item())
            tmp_out = out.detach().cpu()
            preds = torch.argmax(tmp_out, 1)
            # print(label,label.shape)
            prediction = softmax(tmp_out)
        all_out += prediction.numpy().tolist()
        all_score += preds.numpy().tolist()
        all_label += label.detach().cpu().tolist()
        all_label_single +=label_single.detach().cpu().tolist()

    auc = roc_auc_score(all_label, all_out,multi_class='ovo')*100
    acc = accuracy_score(all_label_single, all_score)*100
    AP = []
    all_label = np.array(all_label)
    all_out = np.array(all_out)
    for i in range(len(all_label[0])):
        AP.append(average_precision_score(all_label[:,i],all_out[:,i]))
    f1 = f1_score(all_label_single, all_score, average='macro')*100
    return {'loss': np.mean(all_loss), 'acc':acc,'auc': auc,'f1':f1,'mAP':np.mean(AP)*100}

def train_test(config, model, model_path, train_loader, valid_loader):

    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, config.patience, 0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [config.lr_decay_epoch, config.lr_decay_epoch+25] , gamma= 0.1)    
    current_date = datetime.now()
    folder_name = current_date.strftime('%m.%d')
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    result_path = 'results/{}/{}'.format(config.dataset,folder_name)
    create_folder_if_not_exists(result_path)
    results_file = result_path + '/{}_lr{}_acc_seed{}.tsv'.format(config.modal,config.lr,str(config.random_seed))

    with open(results_file, 'a+', newline='', encoding='utf-8') as tsv_f:
        tsv_w = csv.writer(tsv_f, delimiter='\t')
        tsv_w.writerow(['epoch','train_loss','train_acc','test_loss','test_acc','test_auc','test_f1','test_mAP'])
        tsv_f.close()   

    for epoch in range(config.epoch):
        model.zero_grad()
        print('Is Train Epoch:{}'.format(epoch))
        train_loss, train_acc = train_epoch(config, model, criterion, optimizer, train_loader,scheduler,epoch)
        valid_result = evaluate(config, model, criterion, valid_loader,epoch)
        print('epoch:',epoch,'train_loss:',train_loss,'Train_ACC:',train_acc,'ACC:',valid_result['acc'],'mAP:',valid_result['mAP'])
        with open(results_file, 'a+', newline='', encoding='utf-8') as tsv_f:
            tsv_w = csv.writer(tsv_f, delimiter='\t')
            tsv_w.writerow([epoch,train_loss,train_acc,valid_result['loss'],valid_result['acc'],valid_result['auc'],valid_result['f1'],valid_result['mAP']])
            tsv_f.close() 
            
        
        if valid_result['acc'] > best_acc:
            best_acc=valid_result['acc']
            torch.save(model, model_path)
    model = torch.load(model_path)
    if config.use_cuda:
        model.cuda()
    test_result = evaluate(config, model, criterion, valid_loader,epoch)
    with open(results_file, 'a+', newline='', encoding='utf-8') as tsv_f:
        tsv_w = csv.writer(tsv_f, delimiter='\t')
        tsv_w.writerow(['Accuracy','AUROC','Macro F1','mAP'])
        tsv_w.writerow([test_result['acc'],test_result['auc'],test_result['f1'],test_result['mAP']])
        tsv_f.close()

def run(config):

    set_seed(config.random_seed)
    config.n_gpu = torch.cuda.device_count()
    print("Num GPUs", config.n_gpu)
    print("Device", config.device)
    train_loader,test_loader,_ = get_VA_data_loaders(config)
    model_path = os.path.join('checkpoint',config.dataset, config.our_model+'.pt')
    print('-'*20)
    
    if config.dataset == 'VGGSound':
        config.n_classes = 309
        config.fps=3
    elif config.dataset == 'KS':
        config.n_classes = 31
        config.fps=3
    elif config.dataset == 'CREMA':
        config.n_classes = 6
        config.fps=1
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(config.dataset))
    
    if config.modal == 'audio':
        print('audio')
        model = AudioClassifier(config.n_classes)

    elif config.modal == 'video':
        print('video')
        model = VideoClassifier(config.n_classes,config.fps)   

    folder_path = os.path.join('checkpoint',config.dataset)
    create_folder_if_not_exists(folder_path)
    model.cuda() 
    
    if config.only_test:
        print('Only Test the Model')
        model = torch.load(model_path)
        criterion = nn.CrossEntropyLoss()
        test_result = evaluate(config, model, criterion, test_loader)
        print(test_result['acc'],test_result['auc'],test_result['f1'],test_result['mAP'])
        with open('data/{}/test/{}_acc_seed{}.tsv'.format(config.dataset,config.modal,str(config.random_seed)), 'a+', newline='', encoding='utf-8') as tsv_f:
            tsv_w = csv.writer(tsv_f, delimiter='\t')
            tsv_w.writerow(['Accuracy','AUROC','Macro F1','mAP'])
            tsv_w.writerow([test_result['acc'],test_result['auc'],test_result['f1'],test_result['mAP']])
            tsv_f.close()
        exit()

    train_test(config, model, model_path, train_loader, test_loader)
  

if __name__ == '__main__':
    config = parse_options()
    modal_v = [0]*config.epoch
    modal_a = [0]*config.epoch
    config.our_model = '{}_{}_{}_{}'.format(config.modal,config.optimizer,config.lr,config.our_model)
    target_directory =  config.dataset+'/'+config.our_model + config.optimizer
    if os.path.exists(target_directory):
        if os.path.isdir(target_directory):
            shutil.rmtree(target_directory)
            print("文件夹已成功删除。")
        else:
            print("目标不是文件夹。")
    else:
        print("目标目录不存在。")
        
    writer = SummaryWriter(target_directory)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id     
    run(config)


