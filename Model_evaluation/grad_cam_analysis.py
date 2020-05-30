pip install pytorch-gradcam

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
from torchvision.utils import make_grid, save_image
from torchvision import models
densenet121 = models.densenet121(pretrained = True)

from torch.nn.functional import softplus
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch

import numpy as np
import pandas as pd
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
from __future__ import print_function
from __future__ import division
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
import matplotlib.pyplot as plt
# define size variables
num_features = 224**2

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_features):
        super(VariationalAutoencoder, self).__init__()
        
        self.latent_features = latent_features
        densenet121 = models.densenet121(pretrained = True)
        modules=list(densenet121.children())[-2][:-1]

        # resnet152 = models.resnet152(pretrained=True)
        # modules=list(resnet152.children())[:-2]
        # modules[7][-1].bn3.num_features = latent_features
        # modules[7][-1].conv3.out_channels = latent_features
        
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


latent_features = 1024

#net = VariationalAutoencoder(latent_features).cuda()
import torch
import numpy as np
import pandas as pd
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
from __future__ import print_function
from __future__ import division
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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
#from sampler import ImbalancedDatasetSampler

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



feature_extract = True
num_epochs = 10
batch_size = 1
class Mura(Dataset):
    def __init__(self,df,root,phase, transform=None):
        self.df=df
        #if phase=='train':
        #   #rndperm = np.random.permutation(np.arange(0,len(self.df)))[0:30]
        #  self.df = self.df.iloc[29511:29512,:]
        # if phase=='val':
        #   rndperm = np.random.permutation(np.arange(0,len(self.df)))[0:50]
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
        return img,label_anomaly, label_anatomy, img_name
    
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        #transforms.Resize((224,224)),
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

model = VariationalAutoencoder(latent_features).cuda()
#model.load_state_dict(torch.load('/content/DenseNet_epoch_30.pt'))

# original saved file with DataParallel
state_dict = torch.load('/content/DenseNet_epoch_30.pt')
# create new OrderedDict that does not contain module.
from collections import OrderedDict
new_state_dict = OrderedDict()  
for k, v in state_dict.items():
    name = k[7:] # remove module.
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)


names = (
    'Elbow'
    ,'Shoulder'
    ,'Finger'
    ,'Forearm'
    ,'Hand'
    ,'Humerus'
    ,'Wrist'
)
for inputs, labels_anomaly, labels_anatomy, img_name in dataloaders['val']:
    #data = next(iter(dataloaders['val']))
    
    label = np.where(labels_anatomy==1)[1].item()
    output = model(inputs.cuda())
    if (torch.max(output['anatomy'],1)[1].item() != label):
      print('true class: ',img_name, '\npredicted class:', names[torch.max(output['anatomy'],1)[1].item()],'\n')
    # i +=1
    # if i%100 == 0:
    #     print("image", i)

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
def transform_img(img):
  img = cv2.imread(img)
  mean = np.mean(img[0,:,:])
  sd = np.std(img[0,:,:])
  img[0,:,:] = 0.485 + (img[0,:,:] - mean)*(0.229/sd)
  img[1,:,:] = 0.456 + (img[1,:,:] - mean)*(0.224/sd)
  img[2,:,:] = 0.406 + (img[2,:,:] - mean)*(0.225/sd)
  img  = data_transforms['train'](img)
  #img = img.permute(1,2,0)
  img = img.unsqueeze(0)
  return img 

def plot_imgs(imgs, figsize, label):
  plt.figure(figsize = (figsize[0],figsize[1]))
  for i in range(len(imgs)):
      img = transform_img(imgs[i]).cuda()
      configs = [
          dict(model_type='densenet', arch=densenet121, layer_name='features'),
      ]

      for config in configs:
          config['arch'].cuda().eval()

      cams = [
          [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
          for config in configs
      ]
      images = []
      for gradcam, gradcam_pp in cams:
          mask, _ = gradcam(img)
          heatmap, result = visualize_cam(mask, img)

          mask_pp, _ = gradcam_pp(img)
          heatmap_pp, result_pp_densenet = visualize_cam(mask_pp, img)
          
          images.extend([img.cpu(), heatmap, heatmap_pp, result, result_pp_densenet])


      plt.subplot(1,len(imgs),i+1)
      plt.title('Grad-CAM', fontsize = 30)
      plt.imshow(transforms.ToPILImage()(result_pp_densenet))
      plt.tight_layout()

  plt.figure(figsize=(figsize[0],figsize[1]))
  for i in range(len(imgs)): 
      img = transform_img(imgs[i])   
      plt.subplot(2,len(imgs),i+1)
      plt.title(label[i], fontsize = 30)
      plt.imshow(img[0,0,:,:])
      plt.tight_layout()
     