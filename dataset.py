import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from os.path import isfile, isdir
% matplotlib inline

train = pd.read_csv('../input/train.csv',one_hot=True)
test = pd.read_csv('../input/test.csv',one_hot=True)

print(train.shape)

##Place holder

x = tf.placeholder(tf.float32,shape=[None,784])
## VARIABLES

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

##create graph operations

y = tf.matmul(x,W) + b
# loss function
y_true = tf.placeholder(tf.float32,[None,10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))

### Optimizer

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train =optimizer.minimize(cross_entropy)

##Create session

with tf.Session() as sess:
    
    init=tf.global_variables_initializer()
    
    sess.run(init)
    
    for step in range(1000):
        
       batch_x , batch_y = train.next_batch(100)
        
      sess.run(train,feed_dict={x :batch_x,y_true:batch_y})
        
        ## EVALUATE Model
        
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
        
        
    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
    print (sess.run(acc,feed_dict ={x:test.images,y_true:test.labels}))
