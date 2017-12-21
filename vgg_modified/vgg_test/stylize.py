# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import tensorflow as tf
import numpy as np
import os
import cPickle

from sys import stderr
from vgg_m import const_vgg_net, vgg_net
from PIL import Image

try:
    reduce
except NameError:
    from functools import reduce

def unpickle(file):                                                                                                                                     
    fo = open(file, 'rb')                                                                                                                               
    dict = cPickle.load(fo)                                                                                                                             
    fo.close()                                                                                                                                          
    return dict  

def one_hot_cifar10_vec(label):                                                                                                                            
    vec = np.zeros(10)                                                                                                                                    
    vec[label] = 1                                                                                                                                         
    return vec

def load_cifar10_data():                                                                                                                                   
    print ('Load Cifar10 Dataset')                                                                                                                      
    x_all = []                                                                                                                                          
    y_all = []                                                                                                                                          
    for i in range (5):                                                                                                                                    
        d = unpickle("cifar-10-batches-py/data_batch_" + str(i+1))                                                                                      
        x_ = d['data']                                                                                                                              
        y_ = d['labels']                                                                                                                                
        x_all.append(x_)                                                                                                                                
        y_all.append(y_)                                                                                                                                
                                                                                                                                                        
    d = unpickle('cifar-10-batches-py/test_batch')                                                                                                      
    x_all.append(d['data'])                                                                                                                             
    y_all.append(d['labels'])                                                                                                                           
                                                                                                                                                        
    x = np.concatenate(x_all) / np.float32(255)                                                                                                         
    y = np.concatenate(y_all)                                                                                                                           
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))                                                                                          
    x = x.reshape((x.shape[0], 32, 32, 3))                                                                                                              
                                                                                                                                                    
    pixel_mean = np.mean(x[0:50000],axis=0)                                                                                                             
    x -= pixel_mean                                                                                                                                     
                                                                                                                                                        
    y = map(one_hot_cifar10_vec, y)                                                                                                                        
    X_train = x[0:50000,:,:,:]                                                                                                                          
    Y_train = y[0:50000]                                                                                                                                
    X_test = x[50000:,:,:,:]                                                                                                                            
    Y_test = y[50000:]                                                                                                                                  
                                                                                                                                                    
    return (X_train, Y_train, X_test, Y_test)

X_train, Y_train, X_test, Y_test = load_cifar10_data()
batch_size = 100

g = tf.Graph()
with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    X = tf.placeholder("float", [1, 32, 32, 3])                                                                                                
    Y = tf.placeholder("float", [1, 10]) 

    fc8, prob,layer_list = vgg_net(X)

    saver = tf.train.Saver()
    print 'data_restored'
    saver.restore(sess, './cifar10_progress/')
    
    for k in range(len(layer_list)):
      layer_list[k] = sess.run(layer_list[k], feed_dict={
                X: [X_test[0]],                                                                                                              
                Y: [Y_test[0]]                                                                                                               
          })
    '''
    correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    counts = 0                                                                                                                                          
    cum_acc = 0                                                                                                                                         
    for i in range (0, 10000, batch_size):                                                                                                              
        if i + batch_size < 10000:                                                                                                                      
            acc = sess.run([accuracy],feed_dict={                                                                                                       
                X: X_test[i:i+batch_size],                                                                                                              
                Y: Y_test[i:i+batch_size]                                                                                                               
            })                                                                                                                                          
            counts += 1       
            print 'test#:', counts
            cum_acc += acc[0]                                                                                                                             
    accuracy_now = cum_acc / counts 
    print accuracy_now
    '''

g_net = tf.Graph()
with g_net.as_default():
    X = tf.placeholder("float", [batch_size, 32, 32, 3])                                    
    Y = tf.placeholder("float", [batch_size, 10]) 
    
    fc8,prob = const_vgg_net(X, layer_list)
    
    correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
    
        sess.run(tf.global_variables_initializer())
        counts = 0                                                                                          
        cum_acc = 0                                                                                                                                 
        for i in range (0, 10000, batch_size):                                                                                                      
            if i + batch_size < 10000:                                                              
                acc = sess.run([accuracy],feed_dict={                                                   
                    X: X_test[i:i+batch_size],                                                                              
                    Y: Y_test[i:i+batch_size]                                                                                   
                })                                                                                                                                     
                counts += 1       
                print 'test#:', counts
                cum_acc += acc[0]                                                                                                       
        accuracy_now = cum_acc / counts 
        print accuracy_now

