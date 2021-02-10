
import _pickle
import os
import os.path
import random
import numpy
import tensorflow as tf
import time
import math
import sys
from keras import backend as K

        # if shape is negative, and the runtime is too high, the cdf becomes 1
def ks(runtime, shape, scale, empiric):
    bracket = K.clip(1 + (shape * runtime / scale),0,1000000000)
    eps = 0.001
    res = K.switch(K.less(K.abs(shape), K.ones_like(shape)*eps),1-K.exp(-runtime/scale),1-K.pow(bracket, -(1/shape)))
    #res = 1-K.pow(bracket,-1/shape)
    bool1 = K.greater(runtime,-scale/shape)
    bool2 = K.less(shape,K.zeros_like(shape))
    res = K.switch(tf.logical_and(bool1,bool2),empiric-K.ones_like(empiric),empiric - res)
    return K.abs(res)

def ks_par(y_true, y_pred, c=1.2):
    # runtimes contains a 1xBatchSize vector with the considered runtime for the instance
    # assuming the first space of y_true is the runtime
    # similar for loc scale and shape
    
    
    runtime = K.dot(K.constant([1,0,0,0],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
    empiric = K.dot(K.constant([0,1,0,0],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
    shapePred = K.dot(K.constant([0,0,1,0],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
    scalePred = K.dot(K.constant([0,0,0,1],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
    
    
    scale = K.dot(K.constant([0,1,0,0],dtype = 'float64',shape = (1,4)),K.transpose(y_pred))
    shape = K.dot(K.constant([0,0,1,0],dtype = 'float64',shape = (1,4)),K.transpose(y_pred))
    shape = shape-2
    scale = scale * (10**8)
    
    assert_scale = tf.Assert(tf.reduce_all(tf.logical_not(tf.is_nan(scale))), [y_pred], summarize=16, name ="scale")
    assert_shape = tf.Assert(tf.reduce_all(tf.logical_not(tf.is_nan(shape))), [scale,shape], summarize=16, name="shape")

    with tf.control_dependencies([assert_scale, assert_shape]):
        res = ks(runtime, shape, scale, empiric)-ks(runtime, shapePred, scalePred, empiric)
        shapeRes = shapePred - shape
        #return K.sum(K.abs(res), axis = -1)+c*K.sum(K.abs(shape-shapePred), axis = -1)
        return K.sum(K.abs(res), axis = -1) + c* K.sum(K.abs(shapeRes), axis=-1)
    return K.sum(shape, axis=-1)

def ks_par_modified(y_true, y_pred, c=1.0):
    # runtimes contains a 1xBatchSize vector with the considered runtime for the instance
    # assuming the first space of y_true is the runtime
    # similar for loc scale and shape
    
    
    runtime = K.dot(K.constant([1,0,0,0],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
    empiric = K.dot(K.constant([0,1,0,0],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
    shapePred = K.dot(K.constant([0,0,1,0],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
    scalePred = K.dot(K.constant([0,0,0,1],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
    
    
    scale = K.dot(K.constant([0,1,0,0],dtype = 'float64',shape = (1,4)),K.transpose(y_pred))
    shape = K.dot(K.constant([0,0,1,0],dtype = 'float64',shape = (1,4)),K.transpose(y_pred))
    shape = shape-2
    scale = scale * (10**8)
    
    assert_scale = tf.Assert(tf.reduce_all(tf.logical_not(tf.is_nan(scale))), [y_pred], summarize=16, name ="scale")
    assert_shape = tf.Assert(tf.reduce_all(tf.logical_not(tf.is_nan(shape))), [scale,shape], summarize=16, name="shape")

    with tf.control_dependencies([assert_scale, assert_shape]):  
        res =  ks(runtime,shape, scalePred, empiric)+ks(runtime,shapePred, scale, empiric) #ks(runtime,shape, scale, empiric)+
        return K.sum(res, axis = -1)+c*K.sum(K.abs(shape-shapePred), axis = -1)

    return K.sum(shape, axis=-1)

def make_loss_function(lo=0, c=1.0):
    if lo == 0:
        return lambda y_true, y_pred: ks_par(y_true, y_pred,c=c)
    else:
        return lambda y_true, y_pred: ks_par_modified(y_true, y_pred,c=c)

