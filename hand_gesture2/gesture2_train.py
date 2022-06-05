# 움직이는 제스처 프로그램

import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

actions = ['hi', 'calm down', 'good']

data = np.concatenate([
    np.load('dataset/seq_hi_1654355326.npy'),
    np.load('dataset/seq_calm down_1654355326.npy'),
    np.load('dataset/seq_good_1654355326.npy'),
], axis=0)

print(data.shape)

x_data = data[:, :, :-1]  # 마지막 제외 ㅌ
labels = data[:, 0, :-1]

print(x_data.shape)
print(labels.shape)