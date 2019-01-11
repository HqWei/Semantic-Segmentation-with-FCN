from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from matplotlib import pyplot as plt
import math
import scipy as scp
import scipy.misc
import numpy as np
from skimage import transform,data,io

def data_load(size):
    print('Loading lfw data...')
    base_root = "./data/train/"
    trainx=[]
    trainy=[]
    testx=[]
    testy=[]

    f = open('./data/parts_train.txt')
    file = f.readlines()
    for n in file:
        name1 = n.split()[0]
        name2 = n.split()[1]
        name2 = "%04d" % int(name2)
        name = name1 + '_' + name2
        imt = io.imread('./data/train_64/images/' + name + '.jpg')
        imt1 = io.imread('./data/train_64/labels/' + name + '.ppm')
        trainx.append(imt)
        trainy.append(imt1)
    trainx = np.array(trainx, dtype=np.float32)
    trainy = np.array(trainy, dtype=np.float32)
    # print(trainx.shape)
    # print(trainy.shape)

    f = open('./data/parts_test.txt')
    file = f.readlines()
    for n in file:
        name1 = n.split()[0]
        name2 = n.split()[1]
        name2 = "%04d" % int(name2)
        name = name1 + '_' + name2
        imt = io.imread('./data/test_64/images/' + name + '.jpg')
        # imt= Image.open('/home/hua/Project/face_fcn/data/train_images/' + name + '.jpg')
        imt1 = io.imread('./data/test_64/labels/' + name + '.ppm')
        testx.append(imt)
        testy.append(imt1)
    testx = np.array(testx, dtype=np.float32)
    testy = np.array(testy, dtype=np.float32)
    # print(testx.shape)
    # print(testy.shape)
    return trainx,trainy,testx,testy