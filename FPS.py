
from __future__ import division
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import models as M
import numpy as np
import scipy
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

#time_begin = time.time()

# ===== normalize over the dataset
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized
    
####################################  Load Data #####################################
te_data    = np.load('data_test.npy')
te_mask = np.load('mask_test.npy')
te_mask  = np.expand_dims(te_mask, axis=3)

print('ISIC18 Dataset loaded')

te_data2  = dataset_normalized(te_data)

model = M.Attention_Unet(input_size = (288,384,3))
model.summary()
model.load_weights('weight_isic18')
time_begin = time.time()
predictions = model.predict(te_data, batch_size=4, verbose=1)

predictions = np.where(predictions>0.5, 1, 0)
all_time=time.time()-time_begin
all_time = all_time/predictions.shape[0]
print('FPS is {}'.format(1./all_time))
  






