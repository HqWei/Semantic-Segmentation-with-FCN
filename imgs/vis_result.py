import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

colors=[
    [255,97,0] , #orange
    [0,255,255] , #青色
    [255,0,255] , #粉色
    [255,0,0] #红色
]

def vis_one_result():
    img1 = cv2.imread('result/William_Ford_Jr_0001.jpg', cv2.COLOR_BGR2RGB)
    size = img1.shape
    mask = cv2.imread('result/William_Ford_Jr_0001.png',cv2.IMREAD_GRAYSCALE)
    tmp = np.ones((size[0], size[1], 3), dtype=np.float32)
    weight=0.5
    category = 1
    for i in range(3):
        img1[mask == category, i] = img1[mask == category, i] *weight+tmp[mask == category, i]*(1-weight)*colors[3][i]
    category = 2
    for i in range(3):
        img1[mask == category, i] = img1[mask == category, i] * weight + tmp[mask == category, i] * (1 - weight)* colors[1][i]
    cv2.imshow('dst', img1)
    cv2.imwrite('result.jpg',img1)

vis_one_result()

