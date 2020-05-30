# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:23:15 2020

@author: hp7357
"""
import os 
import numpy as np 
import cv2 
import pandas as pd 

def align_image(img):

		if img.shape[-1] == 3:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		coords = np.column_stack(np.where(img > np.mean(img)))
		angle = cv2.minAreaRect(coords)[-1]

		if angle < -45:
			angle = -(90 + angle)
		else:
			angle = -angle

		# Rotate
		(h, w) = img.shape[:2]
		center = (w // 2, h // 2)
		M = cv2.getRotationMatrix2D(center, angle, 1.0)
		rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

		return rotated, angle 
from scipy import ndimage
n = 10
l = 256

train_path='train/'
valid_path='valid/'

os.chdir('C:/Users/hp7357/Google Drev/Kvantitativ biologi og sygdomsmodellering/Bachelor project/')
os.listdir()

#Load data: 
train_img_path=pd.read_csv('MURA-v1.1/train_image_paths.csv')
valid_img_path=pd.read_csv('MURA-v1.1/valid_image_paths.csv')

for i in valid_img_path.iloc[:,0]:
    image_org = cv2.imread(i)
    rotated_img = align_image(image_org)[0]
    image_org[:,:,0] = rotated_img
    image_org[:,:,1] = rotated_img
    image_org[:,:,2] = rotated_img
    
    gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    if np.mean(gray)< 150:#(np.mean(gray[:10,:])+np.mean(gray[:,:10]))/2
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Find contours, obtain bounding box, extract and save ROI
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = max(cnts, key = cv2.contourArea)
    
    
    
    x,y,w,h = cv2.boundingRect(cnts)
    #cv2.rectangle(image_org, (x, y), (x + w, y + h), (36,255,12), 2)
    if (h*w > 4900) and (h>80) or (w>80):
        ROI = image_org[y:y+h, x:x+w]
    else:
        ROI = image_org
    
    path = i.split('image')[0]
    fullpath = os.path.join(path + i.split('/')[-1].split('.')[0] +'_roi.png')
    cv2.imwrite(fullpath,ROI)
    
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 2))
plt.axes([0, 0, 1, 1])
plt.imshow(ROI)
plt.axis('off')

plt.show()