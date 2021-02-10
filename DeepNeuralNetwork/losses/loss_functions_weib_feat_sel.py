
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
    """ 
        The neural network produces a number Y between 0 and 1. 
        This function convertes this number Y to the logarithm of the associated scale parameter.
    """ 
    mu=  2.863891478217836   
    sigma= 0.146727246649470
    quantile = special_math.ndtri(p)/N.sqrt(2.0)
    return K.exp(mu+sigma*N.sqrt(2.0)*quantile)

def shape_calc(p):
    """
        The neural network produces a number Y between 0 and 1. 
        This function convertes this number Y to the associated shape parameter.
    """
    alpha = 0.968511138239881
    c=17.264902910756167
    k=1.082668010912716
    return alpha*K.pow(K.pow(1/(1-p), 1/k)-1, 1/c)

def ks_weib(y_true, y_pred):
    """
        This function calculates the loss function for the Weibull distribution.
        The loss is based on the KS test statistic.
    
        runtimes contains a 1xBatchSize vector with the considered runtime for the instance
        assuming the first space of y_true is the runtime
        similar for loc scale and shape 
    """


    # The labels contain a measured runtime (in flips; shifted by the 'true' location parameter), 
    # the associated value of the ecdf and the 'true' shape parameter.
    runtime = K.dot(K.constant([1,0,0,0],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
    empiric = K.dot(K.constant([0,1,0,0],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
    shapePred = K.dot(K.constant([0,0,1,0],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
    
    # The NN predicts a scale and a shape parameter for the Weibull distribution.
    scale = K.dot(K.constant([0,1,0,0],dtype = 'float64',shape = (1,4)),K.transpose(y_pred))
    shape = K.dot(K.constant([0,0,1,0],dtype = 'float64',shape = (1,4)),K.transpose(y_pred))
    
    # However, the outputs are scaled to (0,1). 
    # The following steps reverse this scaling to obtain the true estimates of the scale and shape.
    scale = K.exp(scale_calc(scale))
    shape = shape_calc(shape)
    one = K.ones_like(scale)

    # This part of the loss function ensures that the predicted shape approaches the 'true' shape.
    shape_res = K.abs(shape-shapePred)
    
    # Too high shape values may cause numerical issues. Thus, the value is clipped.
    shape = K.clip(shape, 0, 7)
    assert_scale = tf.Assert(tf.reduce_all(tf.logical_not(tf.is_nan(scale))), [y_pred], summarize=16, name ="scale")
    assert_shape = tf.Assert(tf.reduce_all(tf.logical_not(tf.is_nan(shape))), [scale,shape], summarize=16, name="shape")

    with tf.control_dependencies([assert_scale, assert_shape]):
        # This part calculates the value based on the KS test statistic.
        div = runtime / scale
        res = 1-K.exp(-K.pow(div,shape))
        res = (empiric - res)
        result = K.abs(res) + shape_res
        
    return K.sum(result, axis=-1)

def weib_negative_log_like(runtime,scale,shape, tensors = True):
    # deprecated: Not in use anymore
    div = runtime/scale
    
    if tensors:
        return - K.log(shape) + K.log(scale) - (shape-1)*(K.log(runtime)-K.log(scale)) + K.pow(div,shape)
    else:
        return - N.log(shape) + N.log(scale) - (shape-1)*(N.log(runtime)-N.log(scale)) + pow(div,shape)
        

def metric_weib(y_true, y_pred):
    # deprecated: Not in use anymore

    runtime = K.dot(K.constant([1,0,0,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    shapePred = K.dot(K.constant([0,0,1,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    scalePred = K.dot(K.constant([0,0,0,1,0],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    max_nll = K.dot(K.constant([0,0,0,0,1],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    

    scale = K.dot(K.constant([0,1,0,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_pred))
    shape = K.dot(K.constant([0,0,1,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_pred))
 
    scale = scale * (10**8)
    shape_res = shape - shapePred
    pred_nll = weib_negative_log_like(runtime, scale, shape)
    point_max = weib_negative_log_like(runtime, scalePred, shapePred)
    
    result = K.abs((pred_nll-point_max)/point_max)
    result = 2*result +(K.abs(shape_res)/shapePred)
    return K.sum(result, axis=-1)

def ks_log(y_true, y_pred):
    # deprecated: Not in use anymore

    # runtimes contains a 1xBatchSize vector with the considered runtime for the instance
    # assuming the first space of y_true is the runtime
    # similar for loc scale and shape
 
    runtime = K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
    empiric = K.dot(K.constant([0,1,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
    shapePred = K.dot(K.constant([0,0,1],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
 
    scale = K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
    shape = K.dot(K.constant([0,1,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
    scale = scale * 10
 
    percent = K.abs(shape - shapePred)


    res = 0.5 + 0.5*tf.erf((K.log(runtime) - scale)/(N.sqrt(2)*shape))
    res = (empiric - res)
 
    return K.sum(K.abs(res), axis=-1)+K.sum(percent, axis=-1)


def ks_par(y_true, y_pred):
    # deprecated: Not in use anymore

    # runtimes contains a 1xBatchSize vector with the considered runtime for the instance
    # assuming the first space of y_true is the runtime
    # similar for loc scale and shape
    
    runtime = K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
    empiric = K.dot(K.constant([0,1,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
    shapePred = K.dot(K.constant([0,0,1],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
    

    loc =   K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
    scale = K.dot(K.constant([0,1,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
    shape = K.dot(K.constant([0,0,1],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
    
    zero = K.zeros_like(shape)
    one = K.ones_like(shape)
    eps = 0.001 * one
    scale = scale * (10**10)
    shape = shape - 2
    
    cdfPart = 1+K.switch(K.less(shape, zero), K.switch(K.less(runtime, -scale/shape), (shape*runtime)/scale, zero),(shape*runtime)/scale)
    res = K.switch(K.less(K.abs(shape), eps),1-K.exp(-runtime/scale),1-K.pow(cdfPart, -(1/shape)))
    
    return K.sum(K.abs(empiric - res), axis = -1)+K.sum(K.abs(shape-shapePred), axis = -1)
    

def nll_weib(y_true, y_pred):
    # deprecated: Not in use anymore
    
    pun = 0
    K.set_epsilon(1e-12)
    c = K.exp(K.clip(K.abs(y_pred),0,10))*100
    res=0
    punishment = c * K.ones_like(y_pred)
    
    
    runtime = K.clip(K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true)),K.epsilon(),sys.float_info.max) 
    booltensor = K.cast(K.less(y_pred,K.zeros_like(y_pred)),dtype='float64')
    pun = K.sum(K.sum(booltensor * punishment,axis = -1),axis=-1)
    
    
    loc = K.clip(K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred)),K.epsilon(),sys.float_info.max)
    scale = K.clip(K.dot(K.constant([0,1,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred)),K.epsilon(),sys.float_info.max)
    shape = K.clip(K.dot(K.constant([0,0,1],dtype = 'float64',shape = (1,3)),K.transpose(y_pred)),K.epsilon(),sys.float_info.max)
    
    pun = pun + runtime/K.exp(scale)
    
    
    # now, calculate the log of the density
   
    
    one = K.exp(scale)*(runtime)
    two = K.pow(one,shape)
    three = K.exp(-two)
    four = K.pow(one,shape-1) * three
    five =K.clip( four*K.exp(scale)*shape,K.epsilon(),sys.float_info.max)
    six = -K.log(five)
    
    res = K.sum(six, axis = -1)

    
    res = res*pun
    
    return res



def nll_log(y_true, y_pred):
    # deprecated: Not in use anymore

    res=0
    K.set_epsilon(1e-12)


    # runtimes contains a 1xBatchSize vector with the considered runtime for the instance
    # assuming the first space of y_true is the runtime
    # similar for loc scale and shape

    runtime = K.dot(K.constant([1,0],dtype = 'float64',shape = (1,2)),K.transpose(y_true))

    scale = K.dot(K.constant([1,0],dtype = 'float64',shape = (1,2)),K.transpose(y_pred))
    shape = K.dot(K.constant([0,1],dtype = 'float64',shape = (1,2)),K.transpose(y_pred))

    booltensor = K.cast(K.less(shape,K.zeros_like(shape)),dtype='float64')
    c = K.exp(0.5+K.clip(K.abs(shape),0,20))
    punishment = c * K.ones_like(shape)
    pun = booltensor * punishment

    shapeClip = K.clip(shape,K.epsilon(),sys.float_info.max)

    # now to calculate the log of the density

    res = K.square((K.log(runtime)-scale))/(2*K.square(shapeClip)) +(1/2*math.log(2*math.pi)+K.log(shapeClip)+K.log(runtime))
    res = K.switch(K.less(shape,K.zeros_like(shape)),pun,res)

    return K.sum(res,axis = -1)




def nll_par(y_true, y_pred):
    # deprecated: Not in use anymore

    res=0

    # runtimes contains a 1xBatchSize vector with the considered runtime for the instance
    # assuming the first space of y_true is the runtime
    # similar for loc scale and shape

    runtime = K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true))

    location = K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
    scale = K.dot(K.constant([0,1,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
    shape = K.dot(K.constant([0,0,1],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))

    location = K.exp(location)

    booltensor = K.cast(K.less(runtime-location,K.zeros_like(location)),dtype='float64')
    c = K.exp(K.clip(K.abs(runtime-location),0,5))
    punishment = c * K.ones_like(location)
    pun2 = K.sum(K.sum(booltensor * punishment,axis = -1),axis=-1)

    location = K.clip(location,K.epsilon(),sys.float_info.max)
    location = K.minimum(location, runtime-1)

    booltensor = K.cast(K.less((location-K.exp(scale))/(shape),runtime),dtype='float64')
    c = K.exp(K.clip(K.abs(runtime - (location-K.exp(scale))/(shape)),0,5))
    punishment = c * K.ones_like(location)
    pun3 = K.switch(K.less(shape,K.zeros_like(shape)),booltensor * punishment,K.zeros_like(shape))
    pun3 = K.sum(K.sum(pun3, axis = -1), axis = -1)

    runtime = K.switch(K.less(shape,K.zeros_like(shape)), K.minimum((location-K.exp(scale))/(shape) -1, runtime) ,runtime)

    # now to calculate the log of the density

    one = ((1/shape)+K.ones_like(shape))
    two = K.ones_like(shape)+(shape*(runtime-location)/K.exp(scale))
    three = K.log(two)
    res = one * three + scale
    res = K.sum(res,axis = -1) + pun2 + pun3 #+ pun1

    return res
    
