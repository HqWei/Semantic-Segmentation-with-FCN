from matplotlib import pyplot as plt
import math
import scipy as scp
import scipy.misc
import numpy as np
from skimage import transform,data,io

def data_processing():
    f = open('/home/hua/Project/face_fcn/data/parts_train.txt')
    root_path='./data'
    file = f.readlines()
    IMG_SIZE=64
    for n in file:
        name1 = n.split()[0]
        name2 = n.split()[1]
        name2 = "%04d" % int(name2)
        name = name1 + '_' + name2
        imt = io.imread(root_path+'/train/images/' + name + '.jpg')
        imt2 = transform.resize(imt, (IMG_SIZE, IMG_SIZE))
        scipy.misc.imsave(root_path+'/train_64/images/' + name + '.jpg', imt2)
        # imt= Image.open('/home/hua/Project/face_fcn/data/train_images/' + name + '.jpg')
        imt1 = io.imread(root_path+'/train/labels/' + name + '.ppm')
        imt3 = transform.resize(imt1, (IMG_SIZE, IMG_SIZE))
        scipy.misc.imsave(root_path+'/train_64/labels/'+ name + '.ppm', imt3)

    f = open(root_path+'/parts_test.txt')
    file = f.readlines()
    for n in file:
        name1 = n.split()[0]
        name2 = n.split()[1]
        name2 = "%04d" % int(name2)
        name = name1 + '_' + name2
        imt = io.imread(root_path+'/test/images/' + name + '.jpg')
        imt2 = transform.resize(imt, (IMG_SIZE, IMG_SIZE))
        scipy.misc.imsave(root_path+'/test_64/images/' + name + '.jpg', imt2)
        # imt= Image.open('/home/hua/Project/face_fcn/data/train_images/' + name + '.jpg')
        imt1 = io.imread(root_path+'/test/labels/' + name + '.ppm')
        imt3 = transform.resize(imt1, (IMG_SIZE, IMG_SIZE))
        scipy.misc.imsave(root_path+'/test_64/labels/'+ name + '.ppm', imt3)
    # for i in range(328):
    #     imt=io.imread((base_root+'weizmann_horse_db/rgb/horse%03d.jpg')%(i+1))
    #     imt2=transform.resize(imt,(224,224))
    #     scipy.misc.imsave((base_root+'rgb_224/horse%03d.jpg')%(i+1), imt2)
    #
    # for i in range(328):
    #     imt=io.imread((base_root+'weizmann_horse_db/figure_ground/horse%03d.jpg')%(i+1))
    #     imt2=transform.resize(imt,(224,224))
    #     scipy.misc.imsave((base_root+'label_224/horse%03d.jpg')%(i+1), imt2)
if __name__=='__main__':
    data_processing()
