# Loading packages
#-------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import cv2
import time
import copy
import itertools

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader,Dataset
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
from torchvision.transforms import Normalize,CenterCrop,Resize,Compose, ToPILImage

#--------------------------------------------------------------------------------
#os.chdir('/zhome/b8/a/122402/Bachelor/Data/')
os.chdir('../Data')


train_path='train/'
valid_path='valid/'

train_img_path=pd.read_csv('MURA-v1.1/train_image_paths.csv')
valid_img_path=pd.read_csv('MURA-v1.1/valid_image_paths.csv')
train_labels=pd.read_csv('MURA-v1.1/train_labeled_studies.csv')
valid_labels=pd.read_csv('MURA-v1.1/valid_labeled_studies.csv')


# Labels for anomaly
train_img_path['Label']=train_img_path.Img_Path.apply(lambda x:1 if 'positive' in x else 0)
valid_img_path['Label']=valid_img_path.Img_Path.apply(lambda x:1 if 'positive' in x else 0)

#One- hot encoding for anatomy
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

#-----------------------------------------------------------------------------

class Mura(Dataset):
	def __init__(self,df,root,phase, transform=None):
		self.df=df
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

train_mura_dataset=Mura(df=train_img_path,root='./',phase = 'train', transform=data_transforms['train'])
val_mura_dataset=Mura(df=valid_img_path,root='./',phase = 'val',transform=data_transforms['val'])
train_loader=DataLoader(dataset=train_mura_dataset,batch_size= 1,num_workers=0, shuffle = True )
val_loader=DataLoader(dataset=val_mura_dataset,batch_size= 1,num_workers=0, shuffle = True )

dataloaders={
	'train':train_loader,
	'val':val_loader
}
dataset_sizes={
	'train':len(train_mura_dataset),
	'val':len(val_mura_dataset)
}

#----------------------------------------------------------------------------
# Model

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        resnet152 = models.resnet18(pretrained=True)
        modules=list(resnet152.children())[:-2]

        self.encoder = nn.Sequential(*modules)
        

    def forward(self, x): 
        outputs = {}
        x = self.encoder(x)
        x = torch.flatten(x)
       
        return x

net = ResNet()
net.eval()

#feat_cols = [ 'feature'+str(i) for i in range(100352)]
#feat_col = [ 'feature'+str(i) for i in range(100352)]
feat_cols = [ 'feature'+str(i) for i in range(25088)]
feat_col = [ 'feature'+str(i) for i in range(25088)]
feat_cols.append("label")

vis_df = pd.DataFrame(columns = feat_cols) 

for i in range(dataset_sizes['val']):
	data = next(iter(dataloaders['val']))
	
	label = np.where(data[2].type(torch.int16)==1)[1].item()
	
	img_feat = net(data[0]).detach().numpy().flatten()
	img_feat = pd.DataFrame([img_feat], columns = feat_col)
	img_feat['label'] = label
	
	vis_df= pd.concat([vis_df,img_feat], axis = 0)
	
	if i%100 == 0:
		print("image", i)

os.chdir('../PCA')


vis_df.to_pickle('ResNet_vis_df')
