from torch.utils.data import Dataset
import torch

import data_preprocess.preprocess as pp
import numpy as np

from data_preprocess.tokenizer import Tokenizer



class DataGenerator_Sp(Dataset):
    def __init__(self, source_dict, charset, max_text_length, transform):
        self.tokenizer = Tokenizer(charset, max_text_length)
        self.transform = transform
        
        self.dataset = source_dict.copy() 
        
        randomize = np.arange(len(self.dataset['gt']))
        np.random.seed(42)
        np.random.shuffle(randomize)

        self.dataset['dt'] = np.array(self.dataset['dt'])[randomize]
        self.dataset['gt'] = np.array(self.dataset['gt'])[randomize]

        self.dataset['gt'] = [x.decode() for x in self.dataset['gt']]
            
        self.size = len(self.dataset['gt'])

    def __getitem__(self, i):
        img = self.dataset['dt'][i]
    
        img = np.repeat(img[..., np.newaxis], 3, -1)    
        img = pp.normalization(img)
        
        if self.transform is not None:
            img = self.transform(img)

        y_train = self.tokenizer.encode(self.dataset['gt'][i]) 
 
        y_train = np.pad(y_train, (0, self.tokenizer.maxlen - len(y_train)))

        gt = torch.Tensor(y_train)

        return img, gt          

    def __len__(self):
        return self.size
