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
import math
import sys
import acc
VGG_MEAN = [103.939, 116.779, 123.68]


class SNET:#
    def __int__(self):
        self.wd=0.0005
        #self.stddev = 0.001

    def build(self,rgb,num_classes):
        red, green, blue = tf.split(rgb, 3, 3)
        bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], 3)

        self.conv1_1 = self._conv_layer(bgr, "conv1_1",ksize=3,input_chanel=3,output_chanel=16)
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2",ksize=3,input_chanel=16,output_chanel=16)
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name='pool1')

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1",ksize=3,input_chanel=16,output_chanel=32)
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2", ksize=3, input_chanel=32, output_chanel=32)
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name='pool2')

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1", ksize=3, input_chanel=32, output_chanel=64)
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2", ksize=3, input_chanel=64, output_chanel=64)
        self.pool3 = tf.nn.max_pool(self.conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool3')

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1", ksize=3, input_chanel=64, output_chanel=128)
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2", ksize=3, input_chanel=128, output_chanel=128)
        self.pool4 = tf.nn.max_pool(self.conv4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool4')

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1", ksize=3, input_chanel=128, output_chanel=256)
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2", ksize=3, input_chanel=256, output_chanel=256)
        self.pool5 = tf.nn.max_pool(self.conv5_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool5')

        self.score = self._conv_layer(self.pool5, "score", ksize=3, input_chanel=256, output_chanel=num_classes)
        self.upscore4,shape = self._upscore_layer(bottom=self.score,input_chanel=num_classes,output_chanel=num_classes,
                                            shape=tf.shape(bgr),name='upscore4',ksize=64, stride=32)
        self.pred_up = tf.argmax(self.upscore4, dimension=3)

        return self.upscore4,self.pred_up,shape


    def _conv_layer(self, bottom, name,ksize=3,input_chanel=64,output_chanel=64):
        with tf.variable_scope(name) as scope:
            if name=='score':
                kernel = tf.get_variable('weights', shape=[ksize, ksize, input_chanel, output_chanel],
                                         initializer=tf.truncated_normal_initializer(stddev=0.001))
                conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', [output_chanel], initializer=tf.constant_initializer(0.0))
                bias = tf.nn.bias_add(conv, biases)
                conv = bias
            else:
                kernel = tf.get_variable('weights', shape=[ksize, ksize, input_chanel, output_chanel],
                                         initializer=tf.truncated_normal_initializer(stddev=0.001))
                conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.get_variable('biases', [output_chanel], initializer=tf.constant_initializer(0.0))
                bias = tf.nn.bias_add(conv, biases)
                conv = tf.nn.relu(bias, name=scope.name)


        return conv

    def _upscore_layer(self, bottom, input_chanel,output_chanel,shape,
                        name,ksize=4, stride=2):

        strides = [1, stride, stride, 1]

        with tf.variable_scope(name):

            kernel = tf.get_variable('weights', shape=[ksize, ksize, output_chanel,input_chanel],
                                     initializer=tf.truncated_normal_initializer(stddev=0.001))

            new_shape = [shape[0], shape[1], shape[2], output_chanel]
            output_shape = tf.stack(new_shape)
            deconv = tf.nn.conv2d_transpose(bottom, kernel, output_shape,
                                            strides=strides, padding='SAME')

        return deconv,output_shape

def loss(logits, labels, num_classes):
    """
      logits:  [batch_size, width, height, num_classes]. type:float
      labels:  [batch_size, width, height, num_classes]. type:int
    """
    with tf.name_scope('loss'):
        epsilon = tf.constant(value=1e-4) #avoid log0
        logits = tf.reshape(logits, (-1, num_classes))
        labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

        softmax_logits = tf.nn.softmax(logits) + epsilon

        cross_entropy = -tf.reduce_sum(labels * tf.log(softmax_logits), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss



def inference(num_classes,images,labels):

    s_net=SNET()
    upscore,predout,shape=s_net.build(num_classes=num_classes,rgb=images)
    out_loss=loss(logits=upscore,labels=labels,num_classes=num_classes)

    return out_loss,predout,shape