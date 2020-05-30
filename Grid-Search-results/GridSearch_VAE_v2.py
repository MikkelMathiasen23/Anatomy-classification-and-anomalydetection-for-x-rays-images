# Loading packages
#-------------------------------------------------------------------------------
from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import time
import copy
import itertools

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from PIL import Image

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms
from torchvision.transforms import Normalize,CenterCrop,Resize,Compose, ToPILImage

from torch.distributions import Normal, Bernoulli, kl_divergence
from torch.nn.functional import nll_loss
from torch.nn.functional import binary_cross_entropy

from ray import tune


#-------------------------------------------------------------------------------

def train_model(config):
	num_epochs = 10

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
		def __init__(self, z_dim= config['z'], img_channels=1, img_size=224):
			super(VAE_Conv, self).__init__()

			## encoder
			self.encoder = nn.Sequential(
				nn.Conv2d(img_channels, 32, (3,3), stride=2, padding=1),
				nn.ReLU(),
				nn.Conv2d(32, 64, (3,3), stride=2, padding=1),
				nn.ReLU(),
				nn.Conv2d(64, 128, (3,3), stride=2, padding=2),
				nn.ReLU(),
				nn.Conv2d(128, 64, (3,3), stride = 2, padding = 1),
				nn.ReLU(),
				nn.Conv2d(64, 32, (3,3), stride = 2, padding = 1),
				nn.ReLU(),
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
				UnFlatten(32),
				nn.ConvTranspose2d(32, 64, (4,4), stride=2, padding=2),
				nn.ReLU(),
				nn.ConvTranspose2d(64,32, (5,5),stride = 2, padding = 1),
				nn.ReLU(),
				nn.ConvTranspose2d(32, 16, (5,5), stride=2, padding=2),
				nn.ReLU(),
				nn.ConvTranspose2d(16,8, (5,5), stride = 2, padding = 2),
				nn.ReLU(),
				nn.ConvTranspose2d(8,8, (5,5), stride = 2, padding = 2),
				nn.ReLU(),
				nn.ConvTranspose2d(8, img_channels, (4,4), stride=1, padding=2),
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
			outputs['z'] = z
			outputs['log_var'] = logvar
			return outputs
	
	model = VAE_Conv().cuda()

	for name, param in model.named_parameters():
		if param.requires_grad == True:
				print("\t",name)


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

	optimizer = optim.Adam(model.parameters(), lr=0.001)
	loss_function = ELBO_loss

	#-----------------------------------------------------------------------------
	#Setup working directoy
	os.chdir('/zhome/b8/a/122402/Bachelor/Data')


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
	train_loader=DataLoader(dataset=train_mura_dataset,batch_size=64,num_workers=2, shuffle = True )
	val_loader=DataLoader(dataset=val_mura_dataset,batch_size=64,num_workers=2, shuffle = True )

	dataloaders={
		'train':train_loader,
		'val':val_loader
	}
	dataset_sizes={
		'train':len(train_mura_dataset),
		'val':len(val_mura_dataset)
	}

	train_loss, valid_loss = [], []
	train_kl, valid_kl = [], []

	#-------------------------------------------------------------------------------
	for epoch in range(num_epochs):
		print(epoch)
		batch_loss, batch_kl = [], []
		batch_loss_val, batch_kl_val = [], []
		model.train()
		
		# Go through each batch in the training dataset using the loader
		# Note that y is not necessarily known as it is here
		for x, y, _ in dataloaders['train']:
			x = Variable(x[:,0,:,:].unsqueeze(1))
			
			# This is an alternative way of putting
			# a tensor on the GPU
			x = x.cuda()
			
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

		# Evaluate, do not propagate gradients
		with torch.no_grad():
			model.eval()
			for x, y, _ in dataloaders['val']:
				x = Variable(x[:,0,:,:].unsqueeze(1))
			
				x = x.cuda()

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

			tune.track.log(mean_acc =valid_loss)
	#--------------------------------------------------------------------------------------------------

analysis = tune.run(train_model, config={"z": tune.grid_search([200, 400, 600, 800])}, resources_per_trial={'gpu':2})

df = analysis.dataframe()
print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

print(df)
