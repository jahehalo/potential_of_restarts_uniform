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

def ks(x, shape, scale, emp):
    res = 0.5 + 0.5*tf.erf((K.log(x) - scale)/(N.sqrt(2)*shape))
    return K.abs(res - emp)

def ks_log(y_true, y_pred):

 
    runtime = K.dot(K.constant([1,0,0,0],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
    empiric = K.dot(K.constant([0,1,0,0],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
    shapePred = K.dot(K.constant([0,0,1,0],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
    scalePred = K.dot(K.constant([0,0,1,0],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
 
    scale = K.dot(K.constant([1,0,0,0],dtype = 'float64',shape = (1,4)),K.transpose(y_pred))
    shape = K.dot(K.constant([0,1,0,0],dtype = 'float64',shape = (1,4)),K.transpose(y_pred))
 
    scale = scale * 10
 
    percent = K.abs(shape - shapePred)


    res = ks(runtime, shape, scale, empiric)

    return K.sum(res, axis=-1)+1.2*K.sum(percent, axis=-1)# + K.sum(K.abs(shape-shapePred), axis = -1)
    #
    #res =  ks(runtime,shape, scale, empiric)+ks(runtime,shapePred, scale, empiric) + ks(runtime,shape, scalePred, empiric)
    #eturn K.sum(K.abs(res), axis = -1)+K.sum(K.abs(shape-shapePred), axis = -1)
