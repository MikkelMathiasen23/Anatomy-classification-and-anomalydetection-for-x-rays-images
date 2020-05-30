from __future__ import print_function
from __future__ import division
import torch
import numpy as np
import pandas as pd
import os
import cv2
import torch.nn as nn
import torchvision
from torchvision import datasets, models#, transforms
import matplotlib.pyplot as plt
import torch
import torch.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import Normalize,CenterCrop,Resize,Compose, ToPILImage
from torchvision import transforms
from sklearn.decomposition import PCA
import pickle 

#from sampler import ImbalancedDatasetSampler

os.chdir('../Data')

train_path='train/'
valid_path='valid/'

#Load data: 
train_img_path=pd.read_csv('MURA-v1.1/train_image_paths.csv')
valid_img_path=pd.read_csv('MURA-v1.1/valid_image_paths.csv')
train_labels=pd.read_csv('MURA-v1.1/train_labeled_studies.csv')
valid_labels=pd.read_csv('MURA-v1.1/valid_labeled_studies.csv')

#Label anomaly
train_img_path['Label']=train_img_path.Img_Path.apply(lambda x:1 if 'positive' in x else 0)
valid_img_path['Label']=valid_img_path.Img_Path.apply(lambda x:1 if 'positive' in x else 0)

#Label one-hot-encode for anatomy:
train_img_path['ELBOW']= train_img_path.Img_Path.apply(lambda x:1 if 'ELBOW' in x else 0)
train_img_path['SHOULDER']= train_img_path.Img_Path.apply(lambda x:1 if 'SHOULDER' in x else 0)
train_img_path['FINGER']= train_img_path.Img_Path.apply(lambda x:1 if 'FINGER' in x else 0)
train_img_path['FOREARM']= train_img_path.Img_Path.apply(lambda x:1 if 'FOREARM' in x else 0)
train_img_path['HAND']= train_img_path.Img_Path.apply(lambda x:1 if 'HAND' in x else 0)
train_img_path['HUMERUS']= train_img_path.Img_Path.apply(lambda x:1 if 'HUMERUS' in x else 0)
train_img_path['WRIST']= train_img_path.Img_Path.apply(lambda x:1 if 'WRIST' in x else 0)

valid_img_path['ELBOW']= valid_img_path.Img_Path.apply(lambda x:1 if 'ELBOW' in x else 0)
valid_img_path['SHOULDER']= valid_img_path.Img_Path.apply(lambda x:1 if 'SHOULDER' in x else 0)
valid_img_path['FINGER']= valid_img_path.Img_Path.apply(lambda x:1 if 'FINGER' in x else 0)
valid_img_path['FOREARM']= valid_img_path.Img_Path.apply(lambda x:1 if 'FOREARM' in x else 0)
valid_img_path['HAND']= valid_img_path.Img_Path.apply(lambda x:1 if 'HAND' in x else 0)
valid_img_path['HUMERUS']= valid_img_path.Img_Path.apply(lambda x:1 if 'HUMERUS' in x else 0)
valid_img_path['WRIST']= valid_img_path.Img_Path.apply(lambda x:1 if 'WRIST' in x else 0)



batch_size = 1

class Mura(Dataset):
    def __init__(self,df,root,phase, transform=None):
        self.df=df
        #if phase=='train':
        #    rndperm = np.random.permutation(np.arange(0,len(self.df)))[0:1000]
        ##    self.df = self.df.iloc[rndperm,:]
        # if phase=='val':
        #   rndperm = np.random.permutation(np.arange(0,len(self.df)))[0:30]
        #   self.df = self.df.iloc[rndperm,:]
        
        self.root=root
        self.transform=transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        img_name=self.df.iloc[idx,0]
        #img=Image.open(img_name, mode = 'r')
        img = cv2.imread(img_name)
        
        label_anomaly= self.df.iloc[idx,1]
        label_anatomy = np.asarray(self.df.iloc[idx,2:], dtype = np.int16)
        
        if self.transform is not None:
            img=self.transform(img)
        
        # Transformerer image til ImageNet std og Mean 
        # https://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation
        mean = torch.mean(img[0,:,:])
        sd = torch.std(img[0,:,:])
        img[0,:,:] = 0.485 + (img[0,:,:] - mean)*(0.229/sd)
        img[1,:,:] = 0.456 + (img[1,:,:] - mean)*(0.224/sd)
        img[2,:,:] = 0.406 + (img[2,:,:] - mean)*(0.225/sd)
        return img,label_anomaly, label_anatomy
    
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
        #transforms.Normalize([0.236, 0.236, 0.236], [0.109, 0.109, 0.109])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor()
        #transforms.Normalize([0.236, 0.236, 0.236], [0.109, 0.109, 0.109])
    ]),
}

#Dataloader:
train_mura_dataset=Mura(df=train_img_path,root='./',phase = 'train', transform=data_transforms['train'])
val_mura_dataset=Mura(df=valid_img_path,root='./',phase = 'val',transform=data_transforms['val'])
train_loader=DataLoader(dataset=train_mura_dataset,batch_size=batch_size,num_workers=0, shuffle = True )
val_loader=DataLoader(dataset=val_mura_dataset,batch_size=batch_size,num_workers=0, shuffle = True )

dataloaders={
    'train':train_loader,
    'val':val_loader
}
dataset_sizes={
    'train':len(train_mura_dataset),
    'val':len(val_mura_dataset)
}

img_mat = np.zeros((36808,50176))
img_labels_anomaly = np.zeros((36808))
img_labels_anatomy = np.zeros((36808))

i = 0
for inputs, labels_anomaly, labels_anatomy in dataloaders['train']:
    img_labels_anatomy[i] = np.where(labels_anatomy==1)[1].item()
    img_labels_anomaly[i] = labels_anomaly.item()
    img_mat[i,:] = inputs[:,0,:,:].flatten()

    if i%100 ==0:
        print("Image", i)

    i += 1

os.chdir('../PCA')

pca = PCA(n_components = 200)
pca_result = pca.fit_transform(img_mat)

with open('pca_result_raw.pkl','wb') as f:
    pickle.dump(pca_result, f)

#np.savetxt('img_mat', img_mat, delimiter = ',')
np.savetxt('img_labels_anomaly', img_labels_anomaly, delimiter = ',')
np.savetxt('img_labels_anatomy', img_labels_anatomy, delimiter = ',')

"""
feat_cols = [ 'pixel'+str(i) for i in range(50176)]
feat_col = [ 'pixel'+str(i) for i in range(50176)]
feat_cols.append("anatomy")
feat_cols.append('anomaly')

vis_df = pd.DataFrame(columns = feat_cols) 
i =1
for inputs, labels_anomaly, labels_anatomy in dataloaders['train']:
    #data = next(iter(dataloaders['val']))
    
    label = np.where(labels_anatomy==1)[1].item()
    label_2 = labels_anomaly.item()
    img_feat = inputs[:,0,:,:].flatten()#net(inputs).detach().numpy()
    img_feat = pd.DataFrame(img_feat.reshape(-1, img_feat.shape[0]), columns = feat_col)
    img_feat['anatomy'] = label
    img_feat['anomaly'] = label_2
    vis_df= pd.concat([vis_df,img_feat], axis = 0)
    i +=1
    if i%1 == 0:
        print("image", i)

os.chdir('..')

vis_df_2 = vis_df.iloc[:,-2:]
vis_df_2.to_pickle('vis_df_raw_train')
# https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b


feat_col = [ 'pixel'+str(i) for i in range(50176)]

pca = PCA(n_components = 400)
pca_result = pca.fit_transform(vis_df[feat_col].values)

with open('pca_result_raw.pkl','wb') as f:
    pickle.dump(pca_result, f)

"""