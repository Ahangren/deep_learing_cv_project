from sklearn.model_selection import train_test_split
import torch
import pandas as pd

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path='./data/chine'