
import numpy as np


def hist_once(a,b,n):
    k = (a >=0) & (a<n)
    return np.bincount(n*a[k].astype(int) + b[k],minlength=n**2).reshape(n,n)


def acc_eval(eval_images,labels,batch_size,num_class=3):
    nlabels= np.argmax(labels,axis=3)
    hist = np.zeros((num_class,num_class))
    for i in range(batch_size):
        hist += hist_once(eval_images[i].flatten(),nlabels[i].flatten(),num_class)
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print ('overall accuracy:  ',acc)
    return acc


def hist_once2(a,b,n):
    k = (a >=0) & (a<n)
    class_num=np.bincount(a,minlength=11)
    return np.bincount(n*a[k].astype(int) + b[k],minlength=n**2).reshape(n,n),class_num



