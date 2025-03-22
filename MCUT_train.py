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
from models.MaskModel import AVClassifier,MetaWeightAVClassifier
from MyHelper import get_VA_data_loaders
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,average_precision_score
from torch.utils.tensorboard import SummaryWriter
import argparse
import itertools
import random
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from torch.nn import Softmax
from tqdm import tqdm
from itertools import islice
sys.path.append('..')
sys.path.append('../..')
def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def parse_options():
    parser = argparse.ArgumentParser(description='video-audio Dataset Config')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='whether to use gpu')
    parser.add_argument('--random_seed', type=int, default=42,help='random seed')
    parser.add_argument('--data_path', type=str, default='/media/php/data/kinetics_sound')
    parser.add_argument('--pre-train', type=bool, default=False)
    parser.add_argument('--only_test', type=int, default=0)
    parser.add_argument('--tqdm', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=70,
                        help='the total number of epoch in the training')
    
    # parser.add_argument('--alpha', type=float, default=)
    parser.add_argument('--num_workers', type=int , default=16)
    parser.add_argument('--modal', type=str, default='multi',choices=['multi'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--modal_bias', type=float, default=1)
    parser.add_argument('--cagrad_c', type=float, default=0.5)
    
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr_decay_epoch', type=int, default=50,
                        help='the number of epoch to decay lr')
    parser.add_argument('--fusion_method', type=str, default='concat')
    parser.add_argument('--dataset',default='KS',type=str,help='dataset name')
    parser.add_argument('--batch-size', default=32, type=int,help='train batchsize')
    parser.add_argument('--test-batch-size', default=16, type=int,help='train batchsize')
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--modulation', type=int, default=1)
    parser.add_argument('--caculate_cosine', type=int, default=0)
    parser.add_argument('--momentum', type=float, default=0.8)
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

def train_epoch(config, model, criterion, optimizer, train_loader,meta_batch,scheduler,epoch,results_ratio_file,coeff_e,coeff_j):

    all_label, all_out, all_loss, all_score, all_label_single = [], [], [], [], []
    if config.tqdm == 1:
        train_tqdm = tqdm(train_loader)
    else:
        train_tqdm = islice(train_loader, len(train_loader) - 1)
    tsv_f = open(results_ratio_file, 'a+', newline='', encoding='utf-8')
    tsv_w = csv.writer(tsv_f, delimiter='\t')
    model.train()
    meta_lr = config.lr
    softmax = nn.Softmax(dim=1)
    train_length = len(train_loader)
    meta_spectrogram, meta_images, meta_label = meta_batch
    for index, batch in enumerate(train_tqdm):
        param_a_grad  = torch.Tensor([]).cuda()
        param_v_grad  = torch.Tensor([]).cuda()
        param_mm_a_grad  = torch.Tensor([]).cuda()
        param_mm_v_grad  = torch.Tensor([]).cuda()
        spectrogram, images, label = batch
        label_single = torch.argmax(label,1)
        meta_label_single = torch.argmax(meta_label,1)
        fast_parameters = list(model.parameters())
        for weight in model.parameters():
            weight.fast = None
        optimizer.zero_grad()
        #step 1
        out= model(spectrogram.unsqueeze(1).float().cuda(), images.float().cuda(), mode = 0)
        with torch.no_grad():
            # out_meta_mm = model(meta_spectrogram.unsqueeze(1).float().cuda(), meta_images.float().cuda(), mode = 0)
            out_meta_mm = out
        loss_mm = criterion(out, label.cuda())
        grad_mm = torch.autograd.grad(loss_mm, fast_parameters, create_graph=False, allow_unused=True)
        for k, (name ,weight) in enumerate(model.named_parameters()):
            if grad_mm[k] is not None:
                # print(name)
                if weight.fast is None:
                    weight.fast = weight - meta_lr * grad_mm[k]
                else:
                    weight.fast = weight.fast - meta_lr * grad_mm[k]
                    
        a_out, v_out= model(spectrogram.unsqueeze(1).float().cuda(), images.float().cuda(), mode = 1)
        with torch.no_grad():
            meta_a_out, meta_v_out = a_out, v_out
            meta_label_single = label_single
        loss_a = criterion(a_out, label.cuda())
        loss_v = criterion(v_out, label.cuda())
        loss_uni = (loss_a + loss_v) 
        grad_uni = torch.autograd.grad(loss_uni, fast_parameters, allow_unused=True)
        
        
        
        with torch.no_grad():
            ensemble_out =  (meta_a_out + meta_v_out) / 2
            score_ensemble = torch.mean(torch.tensor([softmax(ensemble_out)[i][meta_label_single[i]] for i in range(ensemble_out.size(0))]))
            score_joint = torch.mean(torch.tensor([softmax(out_meta_mm)[i][meta_label_single[i]] for i in range(out_meta_mm.size(0))]))
            t_e =  score_ensemble / config.T
            t_j =  score_joint / config.T
            coeff_j_temp = 2 * torch.exp(t_e)/(torch.exp(t_e)+torch.exp(t_j))
            coeff_e = torch.tensor(1)
            coeff_j = coeff_j*(config.momentum) + coeff_j_temp*(1-config.momentum)
        optimizer.zero_grad()
        
        for k, (name ,weight) in enumerate(model.named_parameters()):
            if grad_mm[k] is not None and grad_uni[k] is not None:
                if config.modulation==1:
                    weight.grad = coeff_j*grad_mm[k] + coeff_e*grad_uni[k]
                else:
                    weight.grad = grad_mm[k] + grad_uni[k]
                if config.caculate_cosine == 1:
                    if "audio" in name and "classifier" not in name:
                        param_mm_a_grad = torch.cat((param_mm_a_grad, grad_mm[k].flatten()), 0)
                        param_a_grad = torch.cat((param_a_grad, grad_uni[k].flatten()), 0)
                    elif "video" in name and "classifier" not in name:
                        param_mm_v_grad = torch.cat((param_mm_v_grad, grad_mm[k].flatten()), 0)
                        param_v_grad = torch.cat((param_v_grad, grad_uni[k].flatten()), 0)
            elif grad_mm[k] is None and grad_uni[k] is not None:
                if config.modulation==1:
                    weight.grad = coeff_e*grad_uni[k]
                else:
                    weight.grad = grad_uni[k]
            elif grad_uni[k] is None and grad_mm[k] is not None:
                if config.modulation==1:
                    weight.grad = coeff_j*grad_mm[k]
                else:
                    weight.grad = grad_mm[k]
            else:
                print(k, name)
                
        if config.caculate_cosine == 1:
            cosine_similarity_a = torch.dot(param_a_grad, param_mm_a_grad).item()/ (torch.norm(param_a_grad) * torch.norm(param_mm_a_grad)).item()
            cosine_similarity_v = torch.dot(param_v_grad, param_mm_v_grad).item()/ (torch.norm(param_v_grad) * torch.norm(param_mm_v_grad)).item()
            dot_a = torch.dot(param_a_grad, param_mm_a_grad).item()
            dot_v = torch.dot(param_v_grad, param_mm_v_grad).item()
            norm_a_grad = torch.norm(param_a_grad).item()
            norm_v_grad = torch.norm(param_v_grad).item()
            tsv_w.writerow([epoch * train_length + index, coeff_e.item(), coeff_j.item(), score_ensemble.item(), score_joint.item(),\
                cosine_similarity_a, cosine_similarity_v, dot_a, dot_v, norm_a_grad, norm_v_grad])
        else:
            tsv_w.writerow([epoch * train_length + index, coeff_e.item(), coeff_j.item(), score_ensemble.item(), score_joint.item()])
            
        optimizer.step()
        fast_parameters = list(model.parameters())
        for weight in model.parameters():
            weight.fast = None
        
        temp_out = out.detach().cpu()
        preds = torch.argmax(temp_out, 1)
        all_score += preds.numpy().tolist()
        all_out += out.detach().cpu().numpy().tolist()
        all_label += label
        all_loss.append(loss_mm.item())
        if config.tqdm == 1:
            train_tqdm.set_description('loss_mm: %f, w_ensemble: %f, w_joint: %f, s_ensemble: %f, s_joint: %f' % (np.mean(all_loss),coeff_e.item(),coeff_j.item(),score_ensemble.item(), score_joint.item()))
        all_label_single +=label_single.detach().cpu().tolist()
        
    tsv_f.close()
    
    scheduler.step()
    acc = accuracy_score(all_label_single, all_score)*100
    return np.mean(all_loss), acc,coeff_e,coeff_j

def evaluate(config, model, criterion, valid_loader, epoch, coeff_e, coeff_j):
    def mAP_calculate(all_label, all_out):
        return np.mean([average_precision_score(all_label[:, i], all_out[:, i]) for i in range(len(all_label[0]))]) * 100

    num = [0.0] * config.n_classes
    acc_a, acc_v = [0.0] * config.n_classes, [0.0] * config.n_classes
    model.eval()
    all_label, all_out, all_loss, all_score, all_label_single = [], [], [], [], []
    all_ensemble, all_out_ensemble, all_fuison, all_correct = [], [], [], []
    softmax = Softmax(dim=1)
    valid_tqdm = tqdm(valid_loader) if config.tqdm else valid_loader
    
    for spectrogram, images, label in valid_tqdm:
        with torch.no_grad():
            spectrogram, images, label = spectrogram.unsqueeze(1).float().cuda(), images.float().cuda(), label.cuda()
            label_single = torch.argmax(label,1)
            
            out = model(spectrogram, images, mode=0)
            a_out, v_out = model(spectrogram, images, mode=1)
            tmp_out = out
            preds = torch.argmax(tmp_out, 1)
            prediction = softmax(tmp_out)
            loss_mm = criterion(out, label)
            all_loss.append(loss_mm.item())   
            pred_v, pred_a = softmax(v_out), softmax(a_out)
            ensemble_out = ((pred_v + pred_a)/2)
            ensemble_preds = torch.argmax(ensemble_out, 1)
            fusion_out = (ensemble_out + prediction)
            fusion_preds = torch.argmax(fusion_out, 1)
            correct_out = (ensemble_out*coeff_e + prediction*coeff_j)
            correct_preds = torch.argmax(correct_out, 1)
            
            for i, label_i in enumerate(label_single):
                num[label_i.cpu()] += 1.0
                v_i, a_i = torch.argmax(pred_v[i]).cpu().item(), torch.argmax(pred_a[i]).cpu().item()
                if label_i == v_i: acc_v[label_i] += 1.0
                if label_i == a_i: acc_a[label_i] += 1.0  

            ensemble_prediction = softmax(v_out+a_out)
            all_ensemble += ensemble_preds.detach().cpu().numpy().tolist()
            all_fuison += fusion_preds.detach().cpu().numpy().tolist()
            all_correct += correct_preds.detach().cpu().numpy().tolist()
            all_out += prediction.detach().cpu().numpy().tolist()
            all_out_ensemble += ensemble_prediction.detach().cpu().numpy().tolist()
            all_score += preds.detach().cpu().numpy().tolist()
            all_label += label.detach().cpu().tolist()
            all_label_single += label.argmax(dim=1).detach().cpu().tolist()

    auc = roc_auc_score(all_label, all_out, multi_class='ovo') * 100
    acc = accuracy_score(all_label_single, all_score) * 100
    ensemble_acc = accuracy_score(all_label_single, all_ensemble) * 100
    fusion_acc = accuracy_score(all_label_single, all_fuison) * 100
    correct_acc = accuracy_score(all_label_single, all_correct) * 100
    
    mAP_value = mAP_calculate(np.array(all_label), np.array(all_out))
    mAP_value_ensemble = mAP_calculate(np.array(all_label), np.array(all_out_ensemble))

    f1 = f1_score(all_label_single, all_score, average='macro')*100
    return {'loss_mm': np.mean(all_loss), 'ensemble_acc' : ensemble_acc, 'fusion_acc':fusion_acc,'correct_acc':correct_acc,'v_out':sum(acc_v) / sum(num) * 100,'a_out':sum(acc_a) / sum(num) * 100, 'acc':acc,'auc': auc,  'f1':f1,'mAP':mAP_value,'mAP_ensemble': mAP_value_ensemble}

def train_test(config, model, model_path, train_loader,meta_loader, valid_loader):

    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [config.lr_decay_epoch, config.lr_decay_epoch+25] , gamma= 0.1)  
    coeff_e,coeff_j = 1, 1
    current_date = datetime.now()
    folder_name = current_date.strftime('%m.%d')
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    result_path = 'results/{}/{}'.format(config.dataset,folder_name)
    create_folder_if_not_exists(result_path)
    results_file = result_path + '/meta_{}_T{}_Momen{}_seed{}.tsv'.format(config.our_model,config.T, config.momentum,str(config.random_seed))
    results_ratio_path = 'ratio/{}'.format(config.dataset)
    create_folder_if_not_exists(results_ratio_path)
    results_ratio_file  = results_ratio_path + '/meta_{}_T{}_Momen{}_seed{}.tsv'.format(config.our_model,config.T, config.momentum,str(config.random_seed))
    
    with open(results_file, 'a+', newline='', encoding='utf-8') as tsv_f:
        tsv_w = csv.writer(tsv_f, delimiter='\t')
        tsv_w.writerow(['epoch','train_loss','train_acc','test_loss','test_acc','test_auc','test_f1','test_mAP','ensemble_acc','fusion_acc','correct_acc','test_audio_t','test_video_t'])
        tsv_f.close()   
    # valid_result = evaluate(config, model, criterion, valid_loader, epoch,coeff_e,coeff_j)
    meta_batch = None
    for batch in train_loader:
        meta_batch = batch
    for epoch in range(config.epoch):
        model.zero_grad()
        print('Is Train Epoch:{}'.format(epoch))
        train_loss, train_acc,coeff_e,coeff_j = train_epoch(config, model, criterion, optimizer, train_loader,meta_batch,scheduler,epoch,results_ratio_file,coeff_e,coeff_j)
        valid_result = evaluate(config, model, criterion, valid_loader, epoch,coeff_e,coeff_j)
        print('epoch:',epoch,'train_loss:',train_loss,'Train_ACC:',train_acc,'ACC:',valid_result['acc'],'mAP:',valid_result['mAP'])
        with open(results_file, 'a+', newline='', encoding='utf-8') as tsv_f:
            tsv_w = csv.writer(tsv_f, delimiter='\t')
            tsv_w.writerow([epoch,train_loss,train_acc,valid_result['loss_mm'],valid_result['acc'],valid_result['auc'],valid_result['f1'], valid_result['mAP'],valid_result['ensemble_acc'], valid_result['fusion_acc'], valid_result['correct_acc'] , valid_result['a_out'], valid_result['v_out']])
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
    train_loader,meta_loader,test_loader = get_VA_data_loaders(config)
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
    elif config.dataset == 'AVE':
        config.n_classes = 28
        config.fps=4
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(config.dataset))
    model = MetaWeightAVClassifier(config)
    
    
    folder_path = os.path.join('checkpoint',config.dataset)
    create_folder_if_not_exists(folder_path)
    model.cuda() 

    train_test(config, model, model_path, train_loader, meta_loader, test_loader)
  

if __name__ == '__main__':
    config = parse_options()
    modal_v = [0]*config.epoch
    modal_a = [0]*config.epoch
    config.our_model = '{}_{}_{}_{}_{}'.format(config.modal_bias,config.modal, config.fusion_method, config.lr,config.our_model)
    target_directory =  'Writer/'+config.dataset+'/'+config.our_model + config.optimizer
    if os.path.exists(target_directory):
        if os.path.isdir(target_directory):
            shutil.rmtree(target_directory)
            print("文件夹已成功删除。")
        else:
            print("目标不是文件夹。")
    else:
        print("目标目录不存在。")
        
    # writer = SummaryWriter(target_directory)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id     
    run(config)


