
import _pickle
import os
import os.path
import random
import numpy as N
import tensorflow as tf
import time
import math
import sys
from keras import backend as K
import metrics
from tensorflow.python.ops.distributions import special_math

# order of parameters: (loc),scale,shape

def scale_calc(p):
    mu=  2.686333243803358  
    sigma= 0.090459464555333
    quantile = special_math.ndtri(p)/N.sqrt(2.0)
    return K.exp(mu+sigma*N.sqrt(2.0)*quantile)

def shape_calc(p):
    alpha = 0.912047580559537 
    c=10.966647985287899 
    k=0.705838374748621 
    return alpha*K.pow(K.pow(1/(1-p), 1/k)-1, 1/c)


def ks_log(y_true, y_pred):
 
    # runtimes contains a 1xBatchSize vector with the considered runtime for the instance
    # assuming the first space of y_true is the runtime
    # similar for loc scale and shape
 
 
 
    # TODO: Check if all clippings make sense
 
 
    runtime = K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
    #runtime = K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
    #runtime = K.ones(shape = (1,3), dtype = 'float64')
    empiric = K.dot(K.constant([0,1,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
    shapePred = K.dot(K.constant([0,0,1],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
 
    scale = K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
    shape = K.dot(K.constant([0,1,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
 
    scale = scale_calc(scale)
    shape = shape_calc(shape)
 
    #scale = scale * (10**8)
    #loc = loc * (10**6)
    #pun = K.switch(K.less(runtime,loc), loc/runtime, K.zeros_like(loc))
    #location = K.switch(K.less(runtime,loc), runtime - 1, loc)
    # now to calculate the log of the density
 
    percent = K.abs(shape - shapePred)#/shapePred


    res = 0.5 + 0.5*tf.erf((K.log(runtime) - scale)/(N.sqrt(2)*shape))
    res = (empiric - res)
 
    return K.sum(K.abs(res), axis=-1)+K.sum(percent, axis=-1)# + K.sum(K.abs(shape-shapePred), axis = -1)

