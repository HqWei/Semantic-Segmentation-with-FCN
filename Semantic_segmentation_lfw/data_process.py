from matplotlib import pyplot as plt

import numpy as np

import cv2
import os
from tqdm import tqdm



def convert_ppm2png(source_dir,ppm_img_list,target_dir):

    with open(ppm_img_list) as f:
        lines=f.readlines()
        for i in tqdm(range(len(lines))):

            img_name=lines[i].rstrip().split('/')[-1]
            mask = cv2.imread(os.path.join(source_dir,lines[i].rstrip().replace('./','')))
            size = mask.shape
            label_copy = 255 * np.ones((size[0], size[1]), dtype=np.float32)

            label_copy[mask[:, :, 0] == 255] = 0
            label_copy[mask[:, :, 1] == 255] = 1
            label_copy[mask[:, :, 2] == 255] = 2

            cv2.imwrite(os.path.join(target_dir,img_name.replace('.ppm','.png')), label_copy)

def make_trainvaltest(img_list,label_list,task_list,save_list):
    fn=open(save_list,'w')
    img_list_dict={}
    with open(img_list) as f:
        for line in f.readlines():
            key=line.rstrip().split('/')[-1].split('.')[0]
            img_list_dict.update({key:line.rstrip()})

    img_label_dict={}
    with open(label_list) as f:
        for line in f.readlines():
            key=line.rstrip().split('/')[-1].split('.')[0]
            img_label_dict.update({key:line.rstrip()})

    with open(task_list) as f:
        for line in f.readlines():
            name,num=line.rstrip().split(' ')
            num=int(num)
            key=name+'_'+'%04.d'%num
            fn.write(img_list_dict[key]+' '+img_label_dict[key]+'\n')

if __name__=="__main__":
    source_dir='/data/0-datasets/LFW/'
    ppm_img_list='/data/0-datasets/LFW/lfw_label.lst'
    target_dir='/data/0-datasets/LFW/gt_png'
    
    convert_ppm2png(source_dir,ppm_img_list, target_dir)

    img_list='lfw/sh30/lfw_imgs.lst'
    label_list='lfw/sh30/lfw_label.lst'

    make_trainvaltest(img_list, label_list, task_list='lfw/lists/train.lst',save_list='lfw/trainval/train.lst')
    make_trainvaltest(img_list, label_list, task_list='lfw/lists/test.lst', save_list='lfw/trainval/test.lst')
    make_trainvaltest(img_list, label_list, task_list='lfw/lists/val.lst', save_list='lfw/trainval/val.lst')
