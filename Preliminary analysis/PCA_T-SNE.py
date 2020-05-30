from __future__ import print_function
import time

import numpy as np
import pandas as pd
import os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

os.chdir('../PCA')


vis_df = pd.read_pickle('DenseNet_vis_df')
vis_df.index = range(len(vis_df.index))

feat_col = [ 'pixel'+str(i) for i in range(50176)]

pca = PCA(n_components=200)
pca_result = pca.fit_transform(vis_df[feat_col].values)


result_df = pd.DataFrame()

result_df['pca-one'] = pca_result[:,0]
result_df['pca-two'] = pca_result[:,1] 
result_df['pca-three'] = pca_result[:,2]

result_df = pd.concat([result_df,vis_df['label']], axis = 1)

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

"""
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="label",
    palette=sns.color_palette("hls", 7),
    data=vis_df.loc[:,:],
    legend="full",
    alpha=0.3)
plt.savefig('PCA-2D.png')

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=vis_df.loc[:,:]['pca-one'],
	ys=vis_df.loc[:,:]['pca-two'],
	zs=vis_df.loc[:,:]['pca-three'],
    c= vis_df.loc[:,:]['label'],
	cmap = 'tab10'
)
ax.set_xlabel('PCA-one')
ax.set_ylabel('PCA-two')
ax.set_zlabel('PCA-three')
plt.savefig('PCA-3D.png')
"""

time_start = time.time()
tsne = TSNE(n_components = 2, verbose =0, perplexity = 40, n_iter = 800)
tsne_result = tsne.fit_transform(pca_result)

print('T-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

result_df['t-sne-2d-one'] = tsne_result[:,0]
result_df['t-sne-2d-two'] = tsne_result[:,1]

"""
plt.figure(figsize = (16,10))
sns.scatterplot(
    x="t-sne-2d-one", y="t-sne-2d-two",
    hue="label",
    palette=sns.color_palette("hls", 7),
    data=vis_df.loc[:,:],
    legend="full",
    alpha=0.3)
plt.savefig('t-sne-2D-PCA200.png')
"""

result_df.to_pickle('DenseNet_result_n200_df')

