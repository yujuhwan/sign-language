# 움직이는 제스처 프로그램

import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

actions = ['hi']

data = np.concatenate([
    np.load('dataset/seq_hi_1653922824.npy'),
], axis=0)

data.shape

