import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
#from data_preprocess import perform_span_corruption_nonseg, perform_span_corruption_seg
from torch.utils.data import Dataset, DataLoader

"""
class ContextData(Dataset):
    def __init__(
        self,
        context,
        cnt,
    ):
        self.context = context
        self.cnt = cnt
        self.raw_data_len = len(self.context)
        
    def __len__(self):
        return len(self.context) * self._cnt

    def __getitem__(self, index: int):
        
        if index % len(self.context) == 0:
            print("start {} corruption".format(index//len(self.context)))
        # support repetition of data with each iteration corrupted diffirently
        idx = index % self.raw_data_len
        data_row = self.context.iloc[idx]
        if idx % 50 == 0:
            print(data_row)
        data_x = [data_row['sepal_length'],data_row['sepal_width'],data_row['petal_length'],data_row['petal_width']]
        data_y = data_row['species']
         


class Net(pl.LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
          

iris_data = pd.read_csv('IRIS.csv')
"""



