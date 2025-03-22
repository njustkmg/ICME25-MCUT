import pickle
import random
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os
import torch
from torchvision import transforms
from typing import Tuple
import csv
import torchvision
from torch import Tensor
# import librosa

class VADataset(Dataset):
    '''
       初始化 
       model = 'train 
       config中参数 train_file 训练集csv文件; test_file 测试集csv文件; data_path数据集根目录
       返回
       spectrogram shape 257 X 1004 
       images shape CTHW : 3 X 3 X 256 X 256
       label
    '''
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        if self.config.fusion_method=='mmtm':
            self.use_pre_frame=1
        elif self.mode=='train':
            self.use_pre_frame=3
            # self.use_pre_frame=3
        else:
            self.use_pre_frame=3
        
        '''
        train部分已修改
        '''
        train_video_data,train_audio_data,train_label,train_class = [],[],[],[]
        test_video_data,test_audio_data,test_label,test_class = [],[],[],[]
        weight_video_data,weight_audio_data,weight_label,weight_class = [],[],[],[]
        root = '/media/php/data/KS'
        train_file = os.path.join(root ,'annotations','train.csv')
        data = pd.read_csv(train_file)
        self.labels= data['label']
        self.files = data['youtube_id']
        for i,item in enumerate(self.files):
            video_dir = os.path.join(root, 'train_img','Image-01-FPS',item)
            # audio_dir = os.path.join(root, 'train_wav', item+'.wav')
            audio_dir = os.path.join(root, 'train_wav_pkl', item+'.pkl')
            if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir))>3 :
                train_video_data.append(video_dir)
                train_audio_data.append(audio_dir)
                if self.labels[i] not in train_class: 
                    train_class.append(self.labels[i])
                train_label.append(self.labels[i])
        '''
        test部分已修改
        '''
        test_file = os.path.join(root, 'annotations','test.csv')
        data = pd.read_csv(test_file)
        self.labels= data['label']
        self.files = data['youtube_id']
        for i,item in enumerate(self.files):
            video_dir = os.path.join(root, 'test_img','Image-01-FPS',item)
            # audio_dir = os.path.join(root, 'test_wav', item+'.wav')
            audio_dir = os.path.join(root, 'test_wav_pkl', item+'.pkl')
            
            if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir))>3:
                test_video_data.append(video_dir)
                test_audio_data.append(audio_dir)
                if self.labels[i] not in test_class: 
                    test_class.append(self.labels[i])
                test_label.append(self.labels[i])
        assert len(train_class) == len(test_class)
        '''
        weight 部分已修改
        '''
        # weight_file = os.path.join(root, 'annotations','weight.csv')
        # data = pd.read_csv(weight_file)
        # self.labels= data['label']
        # self.files = data['youtube_id']
        # root = config.data_path
        # for i,item in enumerate(self.files):
        #     video_dir = os.path.join(root, 'train_img','Image-01-FPS',item)
        #     audio_dir = os.path.join(root, 'train_wav', item+'.wav')
        #     if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir))>3 :
        #         weight_video_data.append(video_dir)
        #         weight_audio_data.append(audio_dir)
        #         if self.labels[i] not in weight_class: 
        #             weight_class.append(self.labels[i])
        #         weight_label.append(self.labels[i])
        
        self.classes = train_class

        class_dict = dict(zip(self.classes, range(len(self.classes))))

        if mode == 'train':
            self.video = train_video_data
            self.audio = train_audio_data
            self.label = [class_dict[train_label[idx]] for idx in range(len(train_label))]
        # if mode == 'weight':
        #     self.video = weight_video_data
        #     self.audio = weight_audio_data
        #     self.label = [class_dict[weight_label[idx]] for idx in range(len(weight_label))]
        if mode == 'test':
            self.video = test_video_data
            self.audio = test_audio_data
            self.label = [class_dict[test_label[idx]] for idx in range(len(test_label))]
        if mode == 'meta':
            self.video = train_video_data
            self.audio = train_audio_data
            self.label = [class_dict[train_label[idx]] for idx in range(len(train_label))]
            combined_data = list(zip(self.video, self.audio, self.label))

            # 随机选择 10% 的元素
            num_samples = len(combined_data)
            num_samples_to_select = int(num_samples * 0.05)
            selected_samples = random.sample(combined_data, num_samples_to_select)

            # 将选中的样本拆分回原始的三个数组
            self.video, self.audio, self.label = zip(*selected_samples)

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):

        # audio
        # sample, rate = librosa.load(self.audio[idx], sr=35400, mono=True)
        # if len(sample)==0:
        #     sample = np.array([0])
        # while len(sample)/rate < 10.:
        #     sample = np.tile(sample, 2)
        # start_point = 0
        # new_sample = sample[start_point:start_point+rate*10]
        # new_sample[new_sample > 1.] = 1.
        # new_sample[new_sample < -1.] = -1.
        # spectrogram = librosa.stft(new_sample, n_fft=512, hop_length=353)
        # spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        # spectrogram = torch.tensor(spectrogram)
        spectrogram = pickle.load(open(self.audio[idx], 'rb'))
        
        # print(np.shape(spectrogram))
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.video[idx])
        # print(self.video[idx])
        select_index = np.random.choice(len(image_samples), size=self.use_pre_frame, replace=False)
        select_index.sort()
            # print(select_index)
        images = torch.zeros((self.use_pre_frame, 3, 224, 224))

        for i in range(self.use_pre_frame):
            img = Image.open(os.path.join(self.video[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))
        # label
        one_hot = np.eye(self.config.n_classes)
        one_hot_label = one_hot[self.label[idx]]
        label = torch.FloatTensor(one_hot_label)
        
        return spectrogram, images, label

class CramedDataset(Dataset):
    
    def __init__(self, config, mode='train'):
        self.config = config
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        self.use_pre_frame=1
        self.data_root = '/media/php/data/CREMA'
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}


        self.train_csv = os.path.join(self.data_root,  'annotations/train.csv')
        self.test_csv = os.path.join(self.data_root,  'annotations/test.csv')

        if mode == 'train' or mode == 'meta':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv
        
            

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.data_root, 'wav_pkl',item[0] + '.pkl')
                visual_path = os.path.join(self.data_root, 'Image-01-FPS', item[0])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
                else:
                    continue
        if mode == 'meta':
            combined_data = list(zip(self.image, self.audio, self.label))

            # 随机选择 10% 的元素
            num_samples = len(combined_data)
            num_samples_to_select = int(num_samples * 0.05)
            selected_samples = random.sample(combined_data, num_samples_to_select)

            # 将选中的样本拆分回原始的三个数组
            self.image, self.audio, self.label = zip(*selected_samples)
             


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # audio
        # samples, rate = librosa.load(self.audio[idx], sr=22050)
        # resamples = np.tile(samples, 3)[:22050*3]
        # resamples[resamples > 1.] = 1.
        # resamples[resamples < -1.] = -1.

        # spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        # spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        # spectrogram = torch.tensor(spectrogram)
        spectrogram = pickle.load(open(self.audio[idx], 'rb'))
        #mean = np.mean(spectrogram)
        #std = np.std(spectrogram)
        #spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        if self.mode == 'train' :
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        select_index = np.random.choice(len(image_samples), size=self.use_pre_frame, replace=False)
        
        select_index.sort()
        # select_index = 1
            # print(select_index)
        images = torch.zeros((self.use_pre_frame, 3, 224, 224))
        for i in range(self.use_pre_frame):
            # try:
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            # except:
            #     img = Image.open(os.path.join(self.image[idx], image_samples[0])).convert('RGB')
                # print(self.image[idx])
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))
        # label
        # label = self.label[idx]
        one_hot = np.eye(self.config.n_classes)
        one_hot_label = one_hot[self.label[idx]]
        label = torch.FloatTensor(one_hot_label)
        # print(images.shape,spectrogram.shape)
        # if self.mode == 'meta':
        #     print( label[0])

        return spectrogram, images, label

class AVEDataset(Dataset):

    def __init__(self, config, mode='train'):
        self.config = config
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        classes = []

        self.data_root = '/media/php/data'
        # class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = '/media/php/data/AVE/'
        self.audio_feature_path = '/media/php/data/AVE/Audio-1004-SE'

        self.train_txt = os.path.join(self.data_root, config.dataset + '/trainSet.txt')
        self.test_txt = os.path.join(self.data_root, config.dataset + '/testSet.txt')
        self.val_txt = os.path.join(self.data_root, config.dataset + '/valSet.txt')

        if mode == 'train':
            txt_file = self.train_txt
        elif mode == 'test':
            txt_file = self.test_txt
        else:
            txt_file = self.val_txt

        with open(self.test_txt, 'r') as f1:
            files = f1.readlines()
            for item in files:
                item = item.split('&')
                if item[0] not in classes:
                    classes.append(item[0])
        class_dict = {}
        for i, c in enumerate(classes):
            class_dict[c] = i

        with open(txt_file, 'r') as f2:
            files = f2.readlines()
            for item in files:
                item = item.split('&')
                audio_path = os.path.join(self.audio_feature_path, item[1] + '.pkl')
                visual_path = os.path.join(self.visual_feature_path, 'Image-01-FPS-SE', item[1])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    if audio_path not in self.audio:
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[item[0]])
                else:
                    continue


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        spectrogram = pickle.load(open(self.audio[idx], 'rb'))

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        # select_index = np.random.choice(len(image_samples), size=self.config.num_frame, replace=False)
        # select_index.sort()
        images = torch.zeros((self.config.fps, 3, 224, 224))
        for i in range(self.config.fps):
            # for i, n in enumerate(select_index):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))

        # label
        # label = self.label[idx]
        one_hot = np.eye(self.config.n_classes)
        one_hot_label = one_hot[self.label[idx]]
        label = torch.FloatTensor(one_hot_label)
        
        return spectrogram, images, label

