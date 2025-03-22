
from MyDataset import VADataset,CramedDataset,AVEDataset
from torch.utils.data import DataLoader
import os
from torchvision import transforms
from torch.utils.data.sampler import Sampler
import random
class RandomSamplerWithReplacement(Sampler):
    def __init__(self, data_source, replacement=True):
        self.data_source = data_source
        self.replacement = replacement

    def __iter__(self):
        while True:
            yield from iter(random.choices(self.data_source, k=len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

def get_VA_data_loaders(config):
    if config.dataset == 'CREMA':        
        train_set = CramedDataset(config,mode='train')
        test_set = CramedDataset(config,mode='test')
        meta_set = CramedDataset(config,mode='meta')
        # weight_set = test_set
    elif config.dataset == 'KS':
        train_set = VADataset(config,mode='train')
        meta_set = VADataset(config,mode='weight')
        test_set = VADataset(config,mode='test')
    elif config.dataset == 'AVE':
        train_set = AVEDataset(config,mode='train')
        test_set = AVEDataset(config,mode='test')
    else:
        raise ValueError('no dataset')
    
    train_loader = DataLoader(
        train_set,
        # test_set,
        batch_size=config.batch_size,
        sampler= None,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=True
    )
    sampler = RandomSamplerWithReplacement(meta_set, replacement=True)
    meta_loader = DataLoader(
        meta_set,
        # test_set,
        batch_size=config.batch_size,
        sampler= sampler,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader,meta_loader,test_loader


