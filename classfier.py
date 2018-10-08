import numpy as np
from PIL import Image
from numpy import *
import scipy.signal as signal
from scipy.ndimage import filters
from sklearn.model_selection import train_test_split
from skimage import color
from skimage import io
import mahotas as mh
import sklearn.preprocessing
from mahotas.features import surf
from sklearn import datasets, svm, metrics
import scipy.ndimage
# from scipy import ndimage
from scipy.misc import imread, imresize, imsave, imrotate
import matplotlib.pyplot as plt
from sklearn import datasets

import os
from sklearn import svm
from scipy import misc
from time import time

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

size = 1000
train_folder = datasets.load_files('./train')
train_img = []
valid_img = []
test_img = []
t0 = time()
# print(" Ensure ordering : ")
# print(train_folder.filenames[5:10])
# print(train_folder.target[5:10])
# print(" Ensure no .DS_Store : ")
# print(len(train_folder.target))

# ----------------------------train:load data--------------------------------------------------------
cnt = 1
# train_folder.target = train_folder.target[:size]
for name1 in train_folder.filenames:
    if cnt > size:
        break
    # if name1[-3:] == 'jpg':
    if name1[-5:] != 'Store':
        img1 = imread(name1)
        img1 = imresize(img1, [140, 50])
        rows, cols = img1.shape[:-1]
        img1 = np.int_(img1 > 128) * 255
        img1_from_raw = img1.reshape(-1, rows * cols)
        img1_from_raw = img1_from_raw[0, :]

        train_img.append(img1_from_raw)  #
        cnt += 1

# ----------------------------score:load data--------------------------------------------------------
cnt = 1
score_folder = datasets.load_files('./valid')
score_img = []
for name1 in score_folder.filenames:
    if cnt > size:
        break
    if name1[-5:] != 'Store':
        img1 = imread(name1)
        img1 = imresize(img1, [140, 50])
        rows, cols = img1.shape[:-1]
        img1 = np.int_(img1 > 128) * 255

        img1_from_raw = img1.reshape(-1, rows * cols)
        img1_from_raw = img1_from_raw[0, :]

        score_img.append(img1_from_raw)  #
        cnt += 1

# ----------------------------predict:load data--------------------------------------------------------
valid_names = []
raw_img = []
path = []
cnt = 1
for root, dirs, files in os.walk('./test_result/test_cropped'):
    for num in files:
        if cnt > size:
            break

        if num[-5:] != 'Store':
            p = os.path.join(root, num)
            path.append(p)
            img = imread(os.path.join(root, num))
            raw_img.append(img)
            img1 = imresize(img, [140, 50])
            rows, cols = img1.shape[:-1]
            img1 = np.int_(img1 > 128) * 255
            img1 = img1.reshape(-1, rows * cols)
            img1 = img1[0, :]
            num, png = num.split('.')

            valid_names.append(num)
            valid_img.append(img1)
            cnt += 1
print('  Data loade Computed: {}'.format(time() - t0))

# ----------------------------SPLIT TEST/TRAIN--------------------------------------------------------
X_train = train_img
X_valid = valid_img

y_train = train_folder.target
# ----------------------------PCA--------------------------------------------------------
print('  Computing PCA...')
t0 = time()
pca = PCA(15, whiten=True)
pca.fit(train_img)
X_train = pca.transform(train_img)
X_valid = pca.transform(valid_img)
X_score = pca.transform(score_img)

# print("X_PAC shape", X_train.shape)  # (30, 1)
print(pca.get_params())
print('  PCA Computed: {}'.format(time() - t0))

# -----------------------------svm.SCV(kernal=rbf)---------------------------------------------------------
print("------------------------svm.SCV(kernal=rbf)---------------------------------")
kernel_svm_time = time()
clf = svm.SVC(kernel='rbf', C=1e4, gamma=0.2)

clf.fit(X_train, y_train)

predict_valid = clf.predict(X_valid)

print("Time for RBF = ", time() - kernel_svm_time)

score = clf.score(X_score, score_folder.target)
print(score)



# for i in range(0, len(predict_valid)):
#     if predict_valid[i] == 0:
#         # raw_img[i] = np.fliplr(raw_img[i])
#         save_path = './test_result/pred_negative/' + valid_names[i] + '.jpg'
#         print(save_path)
#         imsave(save_path, raw_img[i])
#     else:
#         save_path = './test_result/pred_positive/' + valid_names[i] + '.jpg'
#         print(save_path)
#         imsave(save_path, raw_img[i])

