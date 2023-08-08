'''
Dataloader for AntiSmash dataset
Andrew Kiruluta, 05/22/2023
'''
import random
import numpy
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
import torch
import torch.nn as nn

class AntiSmashDataset(Dataset):   
  def __init__(self,df_train, df_target,transform=None):
    self.df_train = df_train
    self.df_target = df_target
    self.transform = transform
    
  def __len__(self):
    return len(self.df_train)
   
  def __getitem__(self,idx):
      self.x_1 = self.df_train.loc[idx,"tokens_1"]
      self.x_2 = self.df_train.loc[idx,"tokens_2"]
      self.y = self.df_target.loc[idx,"tokens"]
      self.x_1 = torch.tensor(self.x_1,dtype=torch.long)
      self.x_1 = (self.x_1).flatten()
      self.x_2 = torch.tensor(self.x_2,dtype=torch.long)
      self.x_2 = (self.x_2).flatten()
      self.y = torch.tensor(self.y,dtype=torch.long)
      self.y = (self.y).flatten()
      return self.x_1, self.x_2, self.y