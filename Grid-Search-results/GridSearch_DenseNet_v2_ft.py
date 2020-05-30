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

from ray import tune

#---------------------------------------------------------------------------------

def train_model(config):
	num_epochs = 10
	feature_extract = False

	class VariationalAutoencoder(nn.Module):
		def __init__(self):
			super(VariationalAutoencoder, self).__init__()
			
			densenet121 = models.densenet121(pretrained = True)
			modules=list(densenet121.children())[-2][:-1]
			
			# We encode the data onto the latent space using two linear layers
			self.encoder = nn.Sequential(*modules)

			
			# The latent code must be decoded into the original image
			self.anomaly = nn.Sequential(
				nn.Linear(in_features=50176, out_features=1)
				)
			self.sigmoid = nn.Sigmoid()
			##
			###Bruger nn.Logsoftmax --> nn.NLLoss i stedet for nn.Crossentropy (hvor softmax er indbygget), for at kunne give 
			### probabilities til f1_score når denne skal beregnes. 
			###
			self.Logsoft = nn.LogSoftmax(dim=1)

			self.anatomy = nn.Sequential(
				nn.Linear(in_features = 50176, out_features = 7)
					)

		def forward(self, x): 
			outputs = {}
			x = self.encoder(x)
			
			x = x.view(x.shape[0],-1)
			
			x_anomaly = self.anomaly(x)
			x_anomaly = self.sigmoid(x_anomaly)

			x_anatomy = self.anatomy(x)
			x_anatomy = self.Logsoft(x_anatomy)

			outputs["anomaly"] = x_anomaly
			outputs["anatomy"] = x_anatomy
			
			return outputs

	net = VariationalAutoencoder()
	
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		net = nn.DataParallel(net)
		os.chdir('/zhome/b8/a/122402/Bachelor/saved_models/DenseNet_v2_e50')
		state_dict = torch.load('DenseNet_epoch_50.pt')
		net.load_state_dict(state_dict)
		device = torch.device("cuda:0")

	else:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		net.to(device)

	net.to(device)
	params_to_update = net.parameters()

	print("Params to learn:")
	if feature_extract:
		params_to_update = []
		for name,param in net.named_parameters():
			if (param.requires_grad == True and'encoder' not in name):
				params_to_update.append(param)
				print("\t",name)
	else:
		for name,param in net.named_parameters():
			if param.requires_grad == True:
				print("\t",name)
	
	optimizer = optim.SGD(params_to_update, lr=config['lr'], momentum=0.9)
	
	# Setup working directory
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
			#if phase=='train':
			#   #rndperm = np.random.permutation(np.arange(0,len(self.df)))[0:30]
			#  self.df = self.df.iloc[29511:29512,:]
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
	train_loader=DataLoader(dataset=train_mura_dataset,batch_size = 256,num_workers=0, shuffle = True )
	val_loader=DataLoader(dataset=val_mura_dataset,batch_size= 256,num_workers=0, shuffle = True )

	dataloaders={
		'train':train_loader,
		'val':val_loader
	}
	dataset_sizes={
		'train':len(train_mura_dataset),
		'val':len(val_mura_dataset)
	}
	criterion = nn.NLLLoss()
	criterion_BCE = nn.BCELoss()

	since = time.time()

	val_acc_anomaly_history = []
	val_acc_anatomy_history = []
	
	cmt_anatomy_t = np.zeros((7,7), dtype=int)
	all_preds_anatomy_t= torch.tensor([])
	all_labels_anatomy_t = torch.tensor([])

	cmt_anomaly_t = np.zeros((2,2), dtype=int)
	all_preds_anomaly_t= torch.tensor([])
	all_labels_anomaly_t = torch.tensor([])
	
	cmt_anatomy_v = np.zeros((7,7), dtype=int)
	all_preds_anatomy_v= torch.tensor([])
	all_labels_anatomy_v = torch.tensor([])

	cmt_anomaly_v = np.zeros((2,2), dtype=int)
	all_preds_anomaly_v= torch.tensor([])
	all_labels_anomaly_v = torch.tensor([])

	cmt_anatomy_iter = np.zeros((7,7), dtype = int)
	cmt_anomaly_iter = np.zeros((2,2), dtype = int)

	best_model_wts = copy.deepcopy(net.state_dict())
	best_acc_anatomy = 0.0
	best_acc_anomaly = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				net.train()  # Set model to training mode
			else:
				net.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects_anomaly = 0.0
			running_corrects_anatomy = 0.0
			SE_anomaly = []
			SE_anatomy = []

			os.chdir('/zhome/b8/a/122402/Bachelor/Data')
			# Iterate over data.
			# Iterate over data.
			for inputs, labels_anomaly, labels_anatomy in dataloaders[phase]:
				inputs = inputs.type(torch.FloatTensor).cuda()
				labels_anomaly = labels_anomaly.type(torch.LongTensor).cuda()
				labels_anatomy = labels_anatomy.type(torch.LongTensor).cuda()
				# zero the parameter gradients
				optimizer.zero_grad()
			   
				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = net(inputs)
					loss = criterion_BCE(outputs['anomaly'].view(-1), labels_anomaly.type(torch.cuda.FloatTensor))
					loss += criterion(outputs['anatomy'], torch.max(labels_anatomy,1)[1])

					preds_anomaly = outputs["anomaly"]
					preds_anatomy = torch.max(outputs['anatomy'],1)[1]

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)

				preds_anomaly = preds_anomaly.clone().detach().cpu()
				preds_anomaly = preds_anomaly.view(-1)
				preds_anomaly = preds_anomaly.round().type(torch.int64).cpu()
				
				labels_anomaly = labels_anomaly.clone().detach().cpu()
				
				preds_anatomy = preds_anatomy.clone().detach().cpu()
				labels_anatomy = labels_anatomy.clone().detach().cpu()
				
				cmt_anatomy_iter = confusion_matrix(torch.max(labels_anatomy,1)[1],preds_anatomy)
				cmt_anomaly_iter = confusion_matrix(labels_anomaly,preds_anomaly)

				recall_anatomy = np.diag(cmt_anatomy_iter) / np.sum((cmt_anatomy_iter), axis = 1)
				recall_anatomy[np.isnan(recall_anatomy)]=0
				precision_anatomy = np.diag(cmt_anatomy_iter) / np.sum((cmt_anatomy_iter), axis = 0)
				precision_anatomy[np.isnan(precision_anatomy)]=0
				
				recall_anomaly = np.diag(cmt_anomaly_iter) / np.sum((cmt_anomaly_iter), axis = 1)
				recall_anomaly[np.isnan(recall_anomaly)]=0
				precision_anomaly = np.diag(cmt_anomaly_iter) / np.sum((cmt_anomaly_iter), axis = 0)
				precision_anomaly[np.isnan(precision_anomaly)]=0
				
				epoch_acc_anomaly = 2*np.mean(recall_anomaly)*np.mean(precision_anomaly)/(np.mean(precision_anomaly)+np.mean(recall_anomaly))
				epoch_acc_anatomy = 2*np.mean(recall_anatomy)*np.mean(precision_anatomy)/(np.mean(precision_anatomy)+np.mean(recall_anatomy))
				
				SE_anatomy.append(epoch_acc_anatomy)
				SE_anomaly.append(epoch_acc_anomaly)
				
				#Implementering egen F1-score beregning:
				if phase == 'train':
					all_preds_anomaly_t = torch.cat(
						(all_preds_anomaly_t, preds_anomaly.type(dtype= torch.FloatTensor)),dim=0)
					all_labels_anomaly_t = torch.cat((all_labels_anomaly_t, labels_anomaly.type(dtype= torch.FloatTensor)),dim=0)
					
					all_preds_anatomy_t = torch.cat(
						(all_preds_anatomy_t, preds_anatomy.type(dtype= torch.FloatTensor)),dim=0)
					all_labels_anatomy_t = torch.cat((all_labels_anatomy_t, labels_anatomy.type(dtype = torch.FloatTensor)),dim=0)
				if phase == 'val':
					all_preds_anomaly_v = torch.cat(
						(all_preds_anomaly_v, preds_anomaly.type(dtype= torch.FloatTensor)),dim=0)
					all_labels_anomaly_v = torch.cat((all_labels_anomaly_v, labels_anomaly.type(dtype= torch.FloatTensor)),dim=0)
					
					all_preds_anatomy_v = torch.cat(
						(all_preds_anatomy_v, preds_anatomy.type(dtype= torch.FloatTensor)),dim=0)
					all_labels_anatomy_v = torch.cat((all_labels_anatomy_v, labels_anatomy.type(dtype = torch.FloatTensor)),dim=0)
				
				  
			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			# epoch_acc_anomaly = running_corrects_anomaly*batch_size / len(dataloaders[phase].dataset)
			# epoch_acc_anatomy = running_corrects_anatomy*batch_size / len(dataloaders[phase].dataset)     
			
			#Beregn F1-score komponenter for anatomy:
			if phase == 'train':
				tmp_anatomy = cmt_anatomy_t.copy()
				cmt_anatomy_t = confusion_matrix(torch.max(all_labels_anatomy_t,1)[1],all_preds_anatomy_t)
				tmp_anomaly = cmt_anomaly_t.copy()
				cmt_anomaly_t = confusion_matrix(all_labels_anomaly_t,all_preds_anomaly_t)
				
				cmt_anatomy = cmt_anatomy_t.copy()
				cmt_anomaly = cmt_anomaly_t.copy()
			if phase == 'val':
				tmp_anatomy = cmt_anatomy_v.copy()
				cmt_anatomy_v = confusion_matrix(torch.max(all_labels_anatomy_v,1)[1],all_preds_anatomy_v)
				tmp_anomaly = cmt_anomaly_v.copy()
				cmt_anomaly_v = confusion_matrix(all_labels_anomaly_v,all_preds_anomaly_v)
				
				cmt_anatomy = cmt_anatomy_v.copy()
				cmt_anomaly = cmt_anomaly_v.copy()
			


			recall_anatomy = np.diag(cmt_anatomy-tmp_anatomy) / np.sum((cmt_anatomy-tmp_anatomy), axis = 1)
			precision_anatomy = np.diag(cmt_anatomy-tmp_anatomy) / np.sum((cmt_anatomy-tmp_anatomy), axis = 0)
			precision_anatomy[np.isnan(precision_anatomy)]=0
			recall_anatomy[np.isnan(recall_anatomy)]=0
			recall_anomaly = np.diag(cmt_anomaly-tmp_anomaly) / np.sum((cmt_anomaly-tmp_anomaly), axis = 1)
			precision_anomaly = np.diag(cmt_anomaly-tmp_anomaly) / np.sum((cmt_anomaly-tmp_anomaly), axis = 0)
			precision_anomaly[np.isnan(precision_anomaly)]=0
			recall_anomaly[np.isnan(recall_anomaly)]=0
			epoch_acc_anomaly = 2*np.mean(recall_anomaly)*np.mean(precision_anomaly)/(np.mean(precision_anomaly)+np.mean(recall_anomaly))
			epoch_acc_anatomy = 2*np.mean(recall_anatomy)*np.mean(precision_anatomy)/(np.mean(precision_anatomy)+np.mean(recall_anatomy))

			anomaly_standard_error = np.std(SE_anomaly)/np.sqrt((len(SE_anomaly)))  
			anatomy_standard_error = np.std(SE_anatomy)/np.sqrt((len(SE_anatomy)))

			tune.track.log(mean_acc =(epoch_acc_anomaly + epoch_acc_anatomy)/2)

			#print('{} Loss: {:.4f} F1-score- anomaly: {:.4f} Standard error anomaly: {:.4f} F1-score- anatomy: {:.4f} Standard error anatomy: {:.4f}'.
			#	  format(phase, epoch_loss, epoch_acc_anomaly, anomaly_standard_error , epoch_acc_anatomy, anatomy_standard_error))
			"""
			# deep copy the model
			if phase == 'val' and epoch_acc_anomaly > best_acc_anomaly and epoch_acc_anatomy > best_acc_anatomy:
				best_acc_anomaly = epoch_acc_anomaly
				best_acc_anatomy = epoch_acc_anatomy
				
			if phase == 'val':
				val_acc_anomaly_history.append(epoch_acc_anomaly)
				val_acc_anomaly_history_SE.append(anomaly_standard_error)

				val_acc_anatomy_history.append(epoch_acc_anatomy)
				val_acc_anatomy_history_SE.append(anatomy_standard_error)
				epoch_loss_val.append(epoch_loss)
			
			if phase == 'train':
				#scheduler.step(epoch_acc_anomaly)
				epoch_loss_train.append(epoch_loss)
				train_acc_anomaly_history.append(epoch_acc_anomaly)
				train_acc_anatomy_history.append(epoch_acc_anatomy)
			"""

	#time_elapsed = time.time() - since
	#print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	#print('Best val F1-score anomaly: {:4f}, F1-score anatomy: {:4f}'.format(best_acc_anomaly, best_acc_anatomy))
	

analysis = tune.run(train_model, config={"lr": tune.grid_search([0.01, 0.001, 0.0001, 0.00001])}, resources_per_trial={'gpu':2})

df = analysis.dataframe()
print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

print(df)