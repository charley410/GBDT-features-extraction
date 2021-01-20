# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:00:38 2020

@author: len
"""



import numpy as np
import pandas as pd
import torch
#import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import train_test_split
'''
from gbdt_onehot_ml import data_x_df,y
x = data_x_df
'''

data = pd.read_csv('C:/Users/len/Desktop/论文/pc_gbdt_one.csv')
data = data.sample(frac=1)    #打乱


x = data.iloc[:,1:-1]
y = data.iloc[:,-1]


class Dataset(Dataset):
    def __init__(self, X, Y):
        self.x = X.values
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


class dnn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(966, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 16)
        self.lin4 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 2)
    
    def forward(self,x_in):
        #print(x_in.shape)
        
        x = F.relu(self.lin1(x_in))

        x = F.relu(self.lin2(x))
        
        x = F.relu(self.lin3(x))
        
        x = F.relu(self.lin4(x))
        features = x
        
        x = self.output(x)
        
        x = torch.sigmoid(x)
        return x,features
        

DEVICE=torch.device("cpu")
if torch.cuda.is_available():
        DEVICE=torch.device("cuda")


criterion = nn.CrossEntropyLoss()
model = dnn_model().to(DEVICE)

#学习率
learning_rate = 0.01
#BS 采用full_batch
batch_size = 297
#优化器
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

'''
train_ds = Dataset(x_train, y_train)
test_ds = Dataset(x_test, y_test)
'''
train_ds = Dataset(x, y)
#dataloader加载数据
'''
train_dl = DataLoader(train_ds, batch_size = batch_size,shuffle = False)  #shuffle = TRUE
test_dl = DataLoader(test_ds, batch_size = batch_size,shuffle = False)
'''
train_dl = DataLoader(train_ds, batch_size = batch_size,shuffle = False)  #shuffle = TRUE

#time
model.train()
#训练10轮
TOTAL_EPOCHS=200
#记录损失函数
losses = [];
for epoch in range(TOTAL_EPOCHS):
    for i, (x, y) in enumerate(train_dl):
        x = x.float().to(DEVICE) #输入必须未float类型
        y = y.long().to(DEVICE) #结果标签必须未long类型
        #清零
        optimizer.zero_grad()
        outputs,features = model(x)
        #计算损失函数
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().data.item())
    print ('Epoch : %d/%d,   Loss: %.4f'%(epoch+1, TOTAL_EPOCHS, np.mean(losses)))
    

model.eval()
correct = 0
total = 0
for i,(x, y) in enumerate(train_dl):
    x = x.float().to(DEVICE)
    y = y.long()
    outputs,features_out = model(x)
    _, predicted = torch.max(outputs.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum()
print('准确率: %.4f %%' % (100 * correct / total))



features_final = features_out.detach().numpy()
#print(features_final)

'''
array = tensor.numpy()
# gpu情况下需要如下的操作
array = tensor.cpu().numpy()
'''


print(DEVICE)


















    
    