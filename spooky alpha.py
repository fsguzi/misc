# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:43:04 2017

@author: Administrator
"""

import pandas as pd
import tensorflow as tf
import numpy as np
import sys
import math
%matplotlib inline

train = pd.read_csv(r'E:\data\train.csv',index_col=False)
test = pd.read_csv(r'E:\data\test.csv',index_col=False)
a_ = train.author.unique().tolist()
train['cat'] = train.author.apply(lambda x: a_.index(x))
train.text = train.text.str.strip()
train.text += '\\'
import unidecode
decoder = unidecode.unidecode
train.text = train.text.str.lower().apply(decoder)
test.text = test.text.str.lower().apply(decoder)
text = train.text.sum()
text += test.text.str.lower().sum()
chars = sorted(list(set(text)))
print(chars,len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
input_size = len(chars)

def makemat(textseries,order = 1):
    num_sentence = len(textseries)
    matrix = np.zeros([35,35])
    for n_s in range(num_sentence):
        sentence = textseries.iloc[n_s]
        for char_ind in range(len(sentence)-order):
            a = sentence[char_ind]
            b = sentence[char_ind+order]
            a = char_indices[a]
            b = char_indices[b]
            matrix[a,b] += 1
    return matrix

def makemat_sen(sentence,order=1):
    matrix = np.zeros([35,35])
    
    for char_ind in range(len(sentence)-order):
        a = sentence[char_ind]
        b = sentence[char_ind+order]
        a = char_indices[a]
        b = char_indices[b]
        matrix[a,b] += 1
    return matrix

def makemat_sen_line(sentence,order=1):
    matrix = np.zeros([35,35])
    
    for char_ind in range(len(sentence)-order):
        a = sentence[char_ind]
        b = sentence[char_ind+order]
        a = char_indices[a]
        b = char_indices[b]
        matrix[a,b] += 1
    return matrix.reshape(1,35*35)

def makemat_sen_line_norm(sentence,order=1):
    matrix = np.zeros([35,35])
    
    for char_ind in range(len(sentence)-order):
        a = sentence[char_ind]
        b = sentence[char_ind+order]
        a = char_indices[a]
        b = char_indices[b]
        matrix[a,b] += 1
    
    return normalizemat(matrix).reshape(1,35*35)


def normalizemat(mat):
    return mat/(np.sum(mat,axis=1,keepdims=True)+1e-6)

total_order = 2
num_sen = len(train)
x = np.zeros((num_sen,35*35*total_order))
y = np.zeros((num_sen,))
for n_s in range(num_sen):
    sen = train.iloc[n_s,1]
    label = train.iloc[n_s,3]    
    y[n_s] = label
    for o in range(total_order):
        
        x[n_s,35*35*o:35*35*(o+1)] = makemat_sen_line(sen,order = o+1)
        
sample_size = 2048*6
x_train = x[:sample_size,:]
y_train = y[:sample_size]

x_valid = x[sample_size:,:]
y_valid = y[sample_size:]

tf.reset_default_graph()

input_size = 35*35*total_order
epoch_num = 250
#sample_size = int(len(train)//1.5)
batch_size = 512
hidden_size = 16

#input and label

X = tf.placeholder(tf.float32,shape=[None,input_size])
Y = tf.placeholder(tf.int32,shape=[None,])


W = tf.Variable(tf.random_normal([input_size,hidden_size]))
b = tf.Variable(tf.random_normal([hidden_size])) 

hidden = tf.matmul(X,W)+b
hidden = tf.sigmoid(hidden)

W2 = tf.Variable(tf.random_normal([hidden_size,3]))
b2 = tf.Variable(tf.random_normal([3])) 

logits = tf.matmul(hidden,W2)+b2
prob = tf.nn.softmax(logits)

loss = tf.losses.sparse_softmax_cross_entropy(labels = Y,
                                              logits = logits,
                                              reduction='none')
LOSS = tf.reduce_sum(loss)
LOSS_ = tf.reduce_mean(loss)
      
LOSS_reg = tf.contrib.layers.l2_regularizer(0.001)
session =tf.Session()
session.run(tf.global_variables_initializer())


#epoch=0
l2 = LOSS_reg(W) + LOSS_reg(W2)
train_step = tf.train.GradientDescentOptimizer(1).minimize(LOSS_+l2)
while True:
    loss_epoch_train = 0
    permutation = np.random.permutation(sample_size)
    x_train = x_train[permutation,:]
    y_train = y_train[permutation]
        
    for i in range(math.ceil(sample_size/batch_size)):
        s = i * batch_size
        e = s + batch_size
        x_ = x_train[s:e,:]
        y_ = y_train[s:e]
            
        train_loss,reg_loss,mean_loss = session.run([LOSS,l2,LOSS_],
                      feed_dict={X:x_,Y:y_})
        session.run(train_step,feed_dict={X:x_train,Y:y_train})
        
        loss_epoch_train += train_loss
    print(reg_loss,mean_loss)
    loss_epoch_valid = session.run(LOSS,
                      feed_dict={X:x_valid,Y:y_valid})
        
    print(epoch,loss_epoch_train/sample_size,
          (loss_epoch_valid/(len(train)-sample_size)))
    epoch +=1