import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import StandardScaler


def GM11(x0):
    x1 = x0.cumsum()  # 1-AGO序列
    z1 = (x1[:len(x1) - 1] + x1[1:]) / 2.0  # 紧邻均值生成序列
    z1 = z1.reshape((len(z1), 1))

    B=np.append(-z1,np.ones_like(z1),axis=1)
    Yn=x0[1:].reshape(len(x0)-1,1)

class TimeSeriesDataset(Dataset):
    def __init__(self,features,labels):
        self.features=features
        self.labels=labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx],self.labels[idx]


class RegressionModel(nn.Module):
    def __init__(self,input_size):
        super(RegressionModel,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(input_size,12),
            nn.ReLU(),
            nn.Linear(12,24),
            nn.ReLU(),
            nn.Linear(24,1)
        )

    def forward(self,x):
        return self.net(x)



if __name__ == '__main__':
    data=pd.read_csv('data/data.csv',index_col=0)
    # print(data.head())
    data.index=range(2000,2020)
    for year in [2020,2021,2022]:
        data.loc[year]=None

    features_for_gm=['x3','x5','x7']
    for col in features_for_gm:
        GM11(data[col].loc[range(2000,2020)].values)
























