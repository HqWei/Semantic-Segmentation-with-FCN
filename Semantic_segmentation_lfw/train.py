from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging

import tensorflow as tf
from matplotlib import pyplot as plt
import math
import scipy as scp
import scipy.misc
import numpy as np
from skimage import transform,data,io
import math
import sys
from data_input import data_load
from model.model import inference
from utils.eval import acc_eval
VGG_MEAN = [103.939, 116.779, 123.68]

Model_path = './snashots/models_64_5'
if not os.path.exists(Model_path):
    os.makedirs(Model_path)
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter,learning_rate,num_steps,power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(learning_rate, i_iter,num_steps,power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def training(loss,learning_rate):

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    return train_op

def main():

    with tf.Graph().as_default():
        k = 0
        k2=0
        loss_list=[]
        tloss_list = []
        x_size,y_size = (64,64)
        batch_size=8
        l_r=0.00008#0.00008
        train_steps=10000
        '''
        Labelde face in the wild dataset load:
        '''
        trainx,trainy,testx,testy=data_load(x_size)
        # with tf.InteractiveSession() as sess:
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = False
        # config.gpu_options.per_process_gpu_memory_fraction = 0.3
        sess=tf.InteractiveSession(config=config)

        images_placeholders = tf.placeholder("float", shape=[None, x_size, y_size, 3])
        labels_placeholders = tf.placeholder("float", shape=[None, x_size, y_size, 3])

        out_loss,pred_out,shape= inference(num_classes=3,images=images_placeholders,labels=labels_placeholders)

        train_op = training(out_loss, l_r)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()


        g1=int(math.ceil(len(trainx)/batch_size))
        g2 = int(math.ceil(len(testx) / batch_size))
        for step in range(train_steps):

            input_images = trainx[0+k*batch_size:batch_size+k*batch_size]
            input_labels = trainy[0+k*batch_size:batch_size+k*batch_size]

            k=k+1
            if k==g1:
                k=0

            feed_dict = {images_placeholders: input_images, labels_placeholders: input_labels}
            _,loss_value,m_shape=sess.run([train_op,out_loss,shape],feed_dict=feed_dict)

            loss_list.append(loss_value)
            if step % 10 ==0:
                print('Step %d:  loss = %.2f' % (step,loss_value))

            if step % 2000==0:

                input_imagest = testx[0 + k2 * batch_size:batch_size + k2 * batch_size]
                input_labelst = testy[0 + k2 * batch_size:batch_size + k2 * batch_size]

                k2 = k2 + 1
                if k2 == g2:
                    k2 = 0

                feed_dict2 = {images_placeholders: input_imagest, labels_placeholders: input_labelst}
                loss_valuet,out = sess.run([out_loss,pred_out], feed_dict=feed_dict2)
                tloss_list.append(loss_valuet)
                print('Step %d:            testloss = %.2f' % (step, loss_valuet))
                save_path = saver.save(sess, Model_path+'/pretrained_lstm.ckpt', global_step=step)
                print("saved to %s" % save_path)



        tloss_list.append(loss_valuet)
        print('Step %d:            testloss = %.2f' % (step, loss_valuet))
        plt.figure(1)
        plt.plot(loss_list)
        plt.title('trainning loss')
        plt.show()
        plt.figure(2)
        plt.plot(tloss_list)
        plt.title('testing loss')
        plt.show()

if __name__=='__main__':
    main()
