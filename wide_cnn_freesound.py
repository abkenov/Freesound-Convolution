#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import librosa 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from IPython.display import Audio
import IPython
import numpy as np
import torch


# In[2]:


df = pd.read_csv("data/train_curated.csv")

mlb = MultiLabelBinarizer()

y = mlb.fit_transform(df['labels'].str.split(','))

class_names = mlb.classes_

index = df['fname']


# # Просмотр данных

# In[3]:


directory = 'data/train_curated/'

for i, file_name in enumerate(index[:0]):
    data, rate = librosa.load(directory + file_name, sr=44100)
    plt.figure(figsize=(10, 10))
    plt.subplot(9, 1, i + 1)    
    plt.title(df['labels'][i])
    plt.plot(data)
    plt.show()
    IPython.display.display(Audio(data, rate=rate))


# In[4]:


directory = 'data/train_curated_spectr/'
for i, file_name in enumerate(index[:0]):
    data = np.load(directory + file_name[:-4] + '.npy')   
    plt.figure(figsize=(15, 15))
    plt.subplot(9, 1, i + 1)    
    plt.title(df['labels'][i])
    plt.imshow(data, origin='top')
    plt.show()


# In[5]:


directory_spectr = '/home/ablan/freesound/data/train_curated_spectr/'
X_list = []

for file_name in index:
    log_spectr = np.load(directory_spectr + file_name[:-3] + 'npy')
#     log_spectr = librosa.power_to_db(spectr)
#     log_spectr = (log_spectr - np.mean(log_spectr)) / (np.std(log_spectr) + 0.001)
    X_list.append(log_spectr)
    
X_list


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_list, y, test_size=0.2, random_state=42)
# X_train = torch.unsqueeze(torch.tensor(X_train, dtype=torch.float32), 1)
# X_test = torch.unsqueeze(torch.tensor(X_test, dtype=torch.float32), 1)
# y_train = torch.tensor(y_train, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.float32)


# In[7]:


# for x in X_train[:5]:
#     plt.imshow(x)
#     plt.show()


# # Train

# In[8]:


# https://pytorch.org/docs/stable/data.html

from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    """Audio dataset."""

    def __init__(self, X, y, mixup, aug):
        self.X = X
        self.y = y
        self.width = 128
        self.delta_min = 0.05
        self.delta_max = 0.1
        self.mixup = mixup
        self.aug = aug
        
    def __len__(self):
        return len(self.X)

    def augmentation(self, X):
        if self.aug == False: 
            return X
        
        delta = np.random.randint(self.delta_min*self.width, self.delta_max*self.width)
        delta_left = np.random.randint(0, 128-delta)
        delta_right = delta_left + delta
        X[:, delta_left:delta_right] = X.min()
        X[delta_left:delta_right, :] = X.min()
        
        return X
    
    def __get_single_item__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        
        if X.shape[1] < self.width:
            X = np.pad(X , ((0, 0),(0, self.width - X.shape[1])), 'minimum')
        elif X.shape[1] > self.width:
            left = np.random.randint(0, X.shape[1] - self.width)
            right = left + self.width
            X = X[:, left:right]
        return X, y
    
    def __getitem__(self, idx):
        if self.mixup == False:
            X, y = self.__get_single_item__(idx)
            X = self.augmentation(X)
            X = torch.unsqueeze(torch.tensor(X, dtype=torch.float32), 0)
            y = torch.tensor(y, dtype=torch.float32)
            return X, y
        else:
            idx2 = np.random.randint(len(self))
            mu = np.random.beta(0.5, 0.5)
            X1, y1 = self.__get_single_item__(idx)
            X2, y2 = self.__get_single_item__(idx2)
            X = X1 * mu + X2 *(1 - mu)
            y = y1 * mu + y2 *(1 - mu)
            X = self.augmentation(X)
            X = torch.unsqueeze(torch.tensor(X, dtype=torch.float32), 0)
            y = torch.tensor(y, dtype=torch.float32).detach()
            return X, y
        
train_audio_dataset = AudioDataset(X_train, y_train, mixup=True, aug=True)
train_dataloader = DataLoader(train_audio_dataset, batch_size=256, shuffle=True, num_workers=4)

test_audio_dataset = AudioDataset(X_test, y_test, mixup=False, aug=False)
test_dataloader = DataLoader(test_audio_dataset, batch_size=256, shuffle=False, num_workers=4)


# In[9]:


i = 0

for x, y in train_audio_dataset:
    plt.imshow(x[0])
    plt.show()
    print(x.shape)
    if i==5:
        break
        
    i+=1


# In[10]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1x128x128
        self.conv_1 = nn.Conv2d(1, 2, kernel_size=3, padding=1, stride=1) 
        self.dense1_bn = nn.BatchNorm2d(2)
        # 2x64x64
        self.conv_2 = nn.Conv2d(2, 4, kernel_size=3, padding=1, stride=1)
        self.dense2_bn = nn.BatchNorm2d(4)
        # 4x32x32
        self.conv_3 = nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=1)
        self.dense3_bn = nn.BatchNorm2d(8)
        # 8x16x16
        self.conv_4 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1)
        self.dense4_bn = nn.BatchNorm2d(16)
        # 16x8x8
        self.conv_5 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.dense5_bn = nn.BatchNorm2d(32)
        # 32x4x4
        self.conv_6 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.dense6_bn = nn.BatchNorm2d(64)
        # 64x2x2        

        self.mp = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.lin_1 = nn.Linear(64 * 2 * 2, 200)
        self.lin_2 = nn.Linear(200, 80)
        
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.mp(x)

        x = self.conv_2(x)
        x = self.dense2_bn(x)
        x = F.relu(x)
        x = self.mp(x)
#         print(x.shape)
        
        x = self.conv_3(x)
        x = self.dense3_bn(x)
        x = F.relu(x)
        x = self.mp(x)

#         print(x.shape)
        x = self.conv_4(x)
        x = self.dense4_bn(x)
        x = F.relu(x)
        x = self.mp(x)
        
#         print(x.shape)
        x = self.conv_5(x)
        x = self.dense5_bn(x)
        x = F.relu(x)
        x = self.mp(x)
        
#         print(x.shape)
        x = self.conv_6(x)
        x = self.dense6_bn(x)
        x = F.relu(x)
        x = self.mp(x)
        
#         print(x.shape)
        x = x.view(x.shape[0], -1)
#        print(x.shape)
        x = self.lin_1(x)
        x = F.relu(x)
        x = self.lin_2(x)
        x = F.softmax(x, dim=1)
        return x


# In[11]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_wide_NN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1x128x128
        self.conv_1 = nn.Conv2d(1, 20, kernel_size=3, padding=1, stride=1) 
        self.dense1_bn = nn.BatchNorm2d(20)
        # 20x64x64
        self.conv_2 = nn.Conv2d(20, 40, kernel_size=3, padding=1, stride=1)
        self.dense2_bn = nn.BatchNorm2d(40)
        # 40x32x32
        self.conv_3 = nn.Conv2d(40, 80, kernel_size=3, padding=1, stride=1)
        self.dense3_bn = nn.BatchNorm2d(80)
        # 80x16x16
        self.conv_4 = nn.Conv2d(80, 160, kernel_size=3, padding=1, stride=1)
        self.dense4_bn = nn.BatchNorm2d(160)
        # 160x8x8
        self.conv_5 = nn.Conv2d(160, 320, kernel_size=3, padding=1, stride=1)
        self.dense5_bn = nn.BatchNorm2d(320)
        # 320x4x4
        self.conv_6 = nn.Conv2d(320, 640, kernel_size=3, padding=1, stride=1)
        self.dense6_bn = nn.BatchNorm2d(640)
        # 640x2x2        

        self.mp = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.lin_1 = nn.Linear(640 * 2 * 2, 200)
        self.lin_2 = nn.Linear(200, 80)
        
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.mp(x)

        x = self.conv_2(x)
        x = self.dense2_bn(x)
        x = F.relu(x)
        x = self.mp(x)
#         print(x.shape)
        
        x = self.conv_3(x)
        x = self.dense3_bn(x)
        x = F.relu(x)
        x = self.mp(x)

#         print(x.shape)
        x = self.conv_4(x)
        x = self.dense4_bn(x)
        x = F.relu(x)
        x = self.mp(x)
        
#         print(x.shape)
        x = self.conv_5(x)
        x = self.dense5_bn(x)
        x = F.relu(x)
        x = self.mp(x)
        
#         print(x.shape)
        x = self.conv_6(x)
        x = self.dense6_bn(x)
        x = F.relu(x)
        x = self.mp(x)
        
#         print(x.shape)
        x = x.view(x.shape[0], -1)
#        print(x.shape)
        x = self.lin_1(x)
        x = F.relu(x)
        x = self.lin_2(x)
        x = F.softmax(x, dim=1)
        return x


# In[12]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_wide_dropout_NN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1x128x128
        self.conv_1 = nn.Conv2d(1, 20, kernel_size=3, padding=1, stride=1) 
        self.dense1_bn = nn.BatchNorm2d(20)
        # 20x64x64
        self.conv_2 = nn.Conv2d(20, 40, kernel_size=3, padding=1, stride=1)
        self.dense2_bn = nn.BatchNorm2d(40)
        # 40x32x32
        self.conv_3 = nn.Conv2d(40, 80, kernel_size=3, padding=1, stride=1)
        self.dense3_bn = nn.BatchNorm2d(80)
        # 80x16x16
        self.conv_4 = nn.Conv2d(80, 160, kernel_size=3, padding=1, stride=1)
        self.dense4_bn = nn.BatchNorm2d(160)
        # 160x8x8
        self.conv_5 = nn.Conv2d(160, 320, kernel_size=3, padding=1, stride=1)
        self.dense5_bn = nn.BatchNorm2d(320)
        # 320x4x4
        self.conv_6 = nn.Conv2d(320, 640, kernel_size=3, padding=1, stride=1)
        self.dense6_bn = nn.BatchNorm2d(640)
        # 640x2x2        

        self.mp = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.lin_1 = nn.Linear(640 * 2 * 2, 200)
        self.lin_2 = nn.Linear(200, 80)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.dense1_bn(x)
        x = F.relu(x)
        x = self.mp(x)
#         x = self.dropout(x)
        
        x = self.conv_2(x)
        x = self.dense2_bn(x)
        x = F.relu(x)
        x = self.mp(x)
#         x = self.dropout(x)
        
#         print(x.shape)
        
        x = self.conv_3(x)
        x = self.dense3_bn(x)
        x = F.relu(x)
        x = self.mp(x)
#         x = self.dropout(x)
        
#         print(x.shape)

        x = self.conv_4(x)
        x = self.dense4_bn(x)
        x = F.relu(x)
        x = self.mp(x)
#         x = self.dropout(x)
        
#         print(x.shape)
        x = self.conv_5(x)
        x = self.dense5_bn(x)
        x = F.relu(x)
        x = self.mp(x)
#         x = self.dropout(x)
        
#         print(x.shape)
        x = self.conv_6(x)
        x = self.dense6_bn(x)
        x = F.relu(x)
        x = self.mp(x)
#         x = self.dropout(x)
        
#         print(x.shape)
        x = x.view(x.shape[0], -1)
#        print(x.shape)
        x = self.dropout(x)
        x = self.lin_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin_2(x)
        x = F.softmax(x, dim=1)
        return x


# In[13]:


from tqdm import tqdm_notebook
from torch import optim
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import label_ranking_average_precision_score

device = 'cuda'
#X_train = X_train.to(device)
#y_train = y_train.to(device)
# X_test= X_test.to(device)
#y_test= y_test.to(device).cpu()
lr = 0.00005
model = Conv_wide_NN().to(device)
# model.load_state_dict(torch.load("models/superwideCNN6_datanorm_aug_small.pth"))
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

loss_val = []
precision_val =[]


# In[14]:


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.90)

for epoch in tqdm_notebook(range(600)):
    for X_train_batch, y_train_batch in train_dataloader:
        y_train_pred = model(X_train_batch.to(device))
        optimizer.zero_grad()
        loss = criterion(y_train_pred, y_train_batch.to(device))    
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        loss_val_batch = []
        precision_val_batch = []
        for X_test_batch, y_test_batch in test_dataloader: 
            y_test_pred = model(X_test_batch.to(device)).cpu()
            loss_val_batch.append(loss.item())
            precision_val_batch.append(label_ranking_average_precision_score(y_test_batch, y_test_pred))
        loss_val.append(np.mean(loss_val_batch))
        precision_val.append(np.mean(precision_val_batch))
        scheduler.step(precision_val[-1])
        lr = optimizer.param_groups[0]['lr']
        print("epoch: {0:} precision: {1:.5f}, loss: {2:.5f}, lr: {3:.7f}".format(epoch, precision_val[-1], loss_val[-1], lr))
        
plt.figure(figsize=(16, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_val)
plt.title('loss')
plt.subplot(1, 2, 2)
plt.plot(precision_val)
plt.title('precision')
plt.savefig('log.png')
plt.close()
        
#     if len(precision_val) > 5 and precision_val[-1] < max(precision_val[-6:-1]):
#         lr = lr * 0.9
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr


# In[15]:


pp = 0
for p in list(model.parameters()):
    n = 1
    for s in list(p.size()):
        n = n*s
    pp += n
print(pp)


# In[16]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(16, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_val)
plt.title('loss')
plt.subplot(1, 2, 2)
plt.plot(precision_val)
plt.title('precision')
print(precision_val[-1])
plt.show()


# In[17]:


model


# In[18]:


model.state_dict()


# In[20]:


torch.save(model.state_dict(), "models/superwideCNN6_aug_betamixup.pth")

