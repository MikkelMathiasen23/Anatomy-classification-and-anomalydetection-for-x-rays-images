###
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
from PIL import Image
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models#, transforms
import matplotlib.pyplot as plt
import time
import copy
import torch
import torch.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import Normalize,CenterCrop,Resize,Compose, ToPILImage
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import torch
import torch.nn.functional as F
#from boilr import BaseGenerativeModel
from torch import nn
from torch.distributions import Normal, Bernoulli, kl_divergence
from torch.nn.functional import nll_loss
from torch.nn.functional import binary_cross_entropy






#-------------------------------------------------------------------------------
# Setup working directory
os.chdir('../Data')

#-------------------------------------------------------------------------------


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


#--------------------------------------------------------------------------------
feature_extract = True
num_epochs = 100
batch_size = 64
lr_rate = 0.001
#--------------------------------------------------------------------------------
###############################

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
        
        label_anomaly = torch.zeros(2, dtype = int)
        label_anomaly[self.df.iloc[idx,1].item()] = 1
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


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, n_channels):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels
    def forward(self, input):
        size = int((input.size(1) // self.n_channels)**0.5)
        return input.view(input.size(0), self.n_channels, size, size)


class VAE_Conv(nn.Module):
    """
    https://github.com/vdumoulin/conv_arithmetic
    """
    def __init__(self, z_dim=200, img_channels=3, img_size=224):
        super(VAE_Conv, self).__init__()

        ## encoder
        densenet121 = models.densenet121(pretrained = True)
        modules=list(densenet121.children())[-2][:-1]
        # We encode the data onto the latent space using two linear layers
        self.encoder = nn.Sequential(*modules,
                                     Flatten()
        )
        
        ## output size depends on input image size
        demo_input = torch.ones([1,img_channels,img_size,img_size])
        h_dim = self.encoder(demo_input).shape[1]
        print('h_dim', h_dim)
        ## map to latent z
        # h_dim = convnet_to_dense_size(img_size, encoder_params)
        self.fc11 = nn.Linear(h_dim, z_dim)
        self.fc12 = nn.Linear(h_dim, z_dim)

        ## decoder
        self.fc2 = nn.Linear(z_dim, h_dim)
        self.decoder = nn.Sequential(
            UnFlatten(1024),
            nn.ConvTranspose2d(1024, 64, (5,5), stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32, (5,5),stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (5,5), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,8, (5,5), stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8,8, (5,5), stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, img_channels, (6,6), stride=1, padding=2),
            nn.Sigmoid()
        )



    def encode(self, x):
        h = self.encoder(x)
        return self.fc11(h), self.fc12(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        img = self.decoder(self.fc2(z))
        return img

    def forward(self, x):
        outputs = {}
        mu, logvar = self.encode(x)
        
        z = self.reparameterize(mu, logvar)
        outputs['x_hat'] = self.decode(z)
        outputs['mu'] = mu
        outputs['log_var'] = logvar
        return outputs
model = VAE_Conv()
#model

if torch.cuda.device_count() > 1:
	print("Let's use", torch.cuda.device_count(), "GPUs!")
	model = nn.DataParallel(model)
	device = torch.device("cuda:0")

else:
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)

model.to(device)


#----------------------------------------------------------------------------------
def ELBO_loss(y, t, mu, log_var):
    # Reconstruction error, log[p(x|z)]
    # Sum over features
    likelihood = -binary_cross_entropy(y, t, reduction="none")
    likelihood = likelihood.view(likelihood.size(0), -1).sum(1)

    # Regularization error: 
    # Kulback-Leibler divergence between approximate posterior, q(z|x)
    # and prior p(z) = N(z | mu, sigma*I).
    
    # In the case of the KL-divergence between diagonal covariance Gaussian and 
    # a standard Gaussian, an analytic solution exists. Using this excerts a lower
    # variance estimator of KL(q||p)
    kl = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=1)

    # Combining the two terms in the evidence lower bound objective (ELBO) 
    # mean over batch
    ELBO = torch.mean(likelihood) - torch.mean(kl)
    
    # notice minus sign as we want to maximise ELBO
    return -ELBO, kl.sum()


# define our optimizer
# The Adam optimizer works really well with VAEs.
optimizer = optim.Adam(model.parameters(), lr=lr_rate)
loss_function = ELBO_loss




from torch.autograd import Variable

train_loss, valid_loss = [], []
train_kl, valid_kl = [], []

#-------------------------------------------------------------------------------
for epoch in range(num_epochs):
    batch_loss, batch_kl = [], []
    batch_loss_val, batch_kl_val = [], []
    model.train()
    
    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for x, y, _ in dataloaders['train']:
        x = Variable(x)
        
        # This is an alternative way of putting
        # a tensor on the GPU
        x = x.to(device)
        
        outputs = model(x)
        x_hat = outputs['x_hat']
        mu, log_var = outputs['mu'], outputs['log_var']

        elbo, kl = loss_function(x_hat, x, mu, log_var)
        
        optimizer.zero_grad()
        elbo.backward()
        optimizer.step()
        
        batch_loss.append(elbo.item())
        batch_kl.append(kl.item())

    train_loss.append(np.mean(batch_loss))
    train_kl.append(np.mean(batch_kl))

    print('Epoch {}, Train elbo Loss: {:.4f}'.format(epoch, train_loss[-1]))

    # Evaluate, do not propagate gradients
    with torch.no_grad():
        model.eval()
        for x, y, _ in dataloaders['val']:
            x = Variable(x)
        
            x = x.to(device)

            outputs = model(x)
            x_hat = outputs['x_hat']
            mu, log_var = outputs['mu'], outputs['log_var']
            #z = outputs["z"]

            elbo, kl = loss_function(x_hat, x, mu, log_var)

            batch_loss_val.append(elbo.item())
            batch_kl_val.append(kl.item())
        
        # We save the latent variable and reconstruction for later use
        # we will need them on the CPU to plot
        #x = x.to("cpu")
        #x_hat = x_hat.to("cpu")
        #z = z.detach().to("cpu").numpy()

        valid_loss.append(np.mean(batch_loss_val))
        valid_kl.append(np.mean(batch_kl_val))

        print('Epoch {}, Val elbo Loss: {:.4f}'.format(epoch, valid_loss[-1]))

#--------------------------------------------------------------------------------------------------
# Saving data models parameters
# From Example 2: https://www.programcreek.com/python/example/101175/torch.save
def save_model_all(model, save_dir, model_name, epoch):
	"""
	:param model:  nn model
	:param save_dir: save model direction
	:param model_name:  model name
	:param epoch:  epoch
	:return:  None
	"""
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	save_prefix = os.path.join(save_dir, model_name)
	save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
	print("save all model to {}".format(save_path))
	output = open(save_path, mode="wb")
	torch.save(model.state_dict(), output)
	# torch.save(model.state_dict(), save_path)
	output.close() 

save_model_all(model, '../saved_models/VAE_v2_DenseNet_e100', 'VAE', 100)

# Creating text files and writing data to them
os.chdir('../saved_models/VAE_v2_DenseNet_e100')

with open('valid_loss.txt' ,'w') as f:
	for item in valid_loss:
		f.write("%s\n" % item)
with open('valid_kl.txt' ,'w') as f:
	for item in valid_kl:
		f.write("%s\n" % item)
with open('train_loss.txt' ,'w') as f:
	for item in valid_loss:
		f.write("%s\n" % item)
with open('train_kl.txt' ,'w') as f:
	for item in valid_kl:
		f.write("%s\n" % item)

def _get_latents(loader):
    #dev = self.experiment.device
    z_all = []
    y_anatomy_all = []
    y_anomaly_all = []
    for x, y_anomaly, y_anatomy in tqdm(loader):
        x = x[:,0,:,:].unsqueeze(1).to(device)
        y_anomaly = y_anomaly.to(device)
        y_anatomy = y_anatomy.to(device)
        out = model(x)
        z_all.append(out['mu'].view(x.size(0), -1))   # mean of q(z)
        y_anatomy_all.append(y_anatomy)
        y_anomaly_all.append(y_anomaly)
    z_all = torch.cat(z_all, dim=0)   # (batch, z_dim)
    y_anomaly_all = torch.cat(y_anomaly_all, dim=0)   # (batch,)
    y_anatomy_all = torch.cat(y_anatomy_all, dim=0)   # (batch,)
    return z_all, y_anatomy_all, y_anomaly_all




def _save_latents(path, dataloaders):
    with torch.no_grad():
          model.eval()

          z,  y_anatomy, y_anomaly = _get_latents(dataloaders['train'])
          fname = os.path.join(path, 'train_latents')
          np.savez_compressed(fname, z=z.cpu(), y_anomaly=y_anomaly.cpu(), y_anatomy = y_anatomy.cpu())
          z,  y_anatomy, y_anomaly = _get_latents(dataloaders['val'])
          fname = os.path.join(path, 'test_latents')
          np.savez_compressed(fname, z=z.cpu(), y_anomaly=y_anomaly.cpu(), y_anatomy = y_anatomy.cpu())


_save_latents('../saved_models/VAE_v2_DenseNet_e100',dataloaders)
