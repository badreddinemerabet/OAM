import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class OAMDataset(Dataset):
    def __init__(self, 
                 df,
                 stds=None,
                 transform=None, 
                 target_transform=None):
                     
        self.df = df
        self.transform = transform
        self.target_transform = target_transform
        self.stds = stds

    def __len__(self):
        return len(self.df)

    def x_y_from_idx(self, idx):
        row = self.df.iloc[idx]
        x = Image.open(row['x'])
        y = Image.open(row['y'])
        return x, y

    def __getitem__(self, idx):
        x, y = self.x_y_from_idx(idx)

        for transform in self.transform:
            x = transform(x)
        for target_transform in self.target_transform:
            y = target_transform(y)

        if self.stds:
            x = torch.tile(x, (len(self.stds)+1, 1, 1, 1))
            y = torch.tile(y, (len(self.stds)+1, 1, 1, 1))
            
            for i, std in enumerate(self.stds):
                noise = torch.normal(mean=0.0, 
                                     std=std,
                                     size=y[i].shape)
                
                y[i] = torch.clip(y[i] + noise, 
                                  min=0.0, 
                                  max=1.0)
            
        return x, y

def get_split_df(df_path,
                train_split=0.9,
                **kwargs):

    df = pd.read_csv(df_path)

    n_train = int(train_split * len(df))
    train = df.sample(n_train, replace=False)

    test = df.loc[~df.index.isin(train.index)]

    train_kwargs = kwargs
    test_kwargs = {k:v for k,v in kwargs.items() if k != 'stds'}
                    
    return OAMDataset(train,
                      **train_kwargs), \
            OAMDataset(test, 
                       **test_kwargs)