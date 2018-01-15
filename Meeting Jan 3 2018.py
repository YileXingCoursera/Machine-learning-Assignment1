# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:13:57 2018

@author: 13501
"""

import numpy as np
num_examples=4
nn_input_dim=2
nn_output_dim=1
nn_hdim=2#I can change the number to 4 or 6
alpha=0.1
def loss_function(W1,b1,W2,b2,X,y):
    z1=X.dot(W1)+b1
    a1=np.tanh(z1)
    z2=a1.dot(W2)+b2
    a2=np.tanh(z2)
    difference=a2-y
    difference=difference**2
    loss=(1/2)*np.sum(difference)
    return loss
def predict(W1,b1,W2,b2,X):
    z1=X.dot(W1)+b1
    a1=np.tanh(z1)
    z2=a1.dot(W2)+b2
    a2=np.tanh(z2)
    return a2
np.random.seed(0)
W1=np.random.randn(nn_input_dim,nn_hdim)/np.sqrt(nn_input_dim)
b1=np.zeros((1,nn_hdim))
W2=np.random.randn(nn_hdim,nn_output_dim)/np.sqrt(nn_hdim)
b2=np.zeros((1,nn_output_dim))
loss=loss_function(W1,b1,W2,b2,X[:,0],y,num_examples)
model={}
z1=X[:,0].dot(W1)+b1
a1=np.tanh(z1)
z2=a1.dot(W2)+b2
a2=np.tanh(z2)
delta3=a2-y
dW2=(a1.T).dot(delta3)
db2=np.sum(delta3,axis=0,keepdims=True)
delta2=delta3.dot(W2.T)*(1-np.power(a1,2))
db1=np.sum(delta2,axis=0)
i=0
while (abs(loss)>0.05):
    col=i%4
    dW1=np.dot(X[:,col].T,delta2)
    i=i+1
    W1=W1-alpha*dW1
    b1=b1-alpha*db1
    W2=W2-alpha*dW2
    b2=b2-alpha*db2
    if col==3:
        loss=loss_function(W1,b1,W2,b2,X[:0],y)
    else:
        loss=loss_function(W1,b1,W2,b2,X[:,col+1],y)
    if i % 100 == 0:
        print "Loss after iteration %i: %f" %(i,loss)
model={'W1':W1,'b1':b1,'W2':W2,'b2':b2}
a2=predict(W1,b1,W2,b2,X)