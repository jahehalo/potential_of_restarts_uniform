
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

# order of parameters: (loc),scale,shape

def ks_weib(y_true, y_pred):
    
    # runtimes contains a 1xBatchSize vector with the considered runtime for the instance
    # assuming the first space of y_true is the runtime
    # similar for loc scale and shape
    
    
    # TODO: Check if all clippings make sense
    
    
    runtime = K.dot(K.constant([1,0,0,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    #runtime = K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
    #runtime = K.ones(shape = (1,3), dtype = 'float64')
    empiric = K.dot(K.constant([0,1,0,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    shapePred = K.dot(K.constant([0,0,1,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    
    #loc =   K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
    scale = K.dot(K.constant([0,1,0,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_pred))
    shape = K.dot(K.constant([0,0,1,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_pred))
    scale = scale * (10**8)
    one = K.ones_like(scale)

    shape_res = K.abs(shape-shapePred)
    #shape_res = K.switch(K.less(one, shape_res), shape_res, K.zeros_like(shape))

    shape = K.clip(shape, 0, 7)
    #scale = K.clip(scale,0.001,10**30)
    assert_scale = tf.Assert(tf.reduce_all(tf.logical_not(tf.is_nan(scale))), [y_pred], summarize=16, name ="scale")
    assert_shape = tf.Assert(tf.reduce_all(tf.logical_not(tf.is_nan(shape))), [scale,shape], summarize=16, name="shape")

    with tf.control_dependencies([assert_scale, assert_shape]):
        div = runtime / scale
        res = 1-K.exp(-K.pow(div,shape))
        res = (empiric - res)
        result = K.abs(res) + shape_res
        #result = K.maximum(K.abs(res), K.abs(shape-shapePred))
        #assert_op = tf.Assert(tf.reduce_all(tf.logical_not(tf.logical_or(tf.is_nan(result),tf.is_inf(result)))), [scale, shape, runtime])
        #with tf.control_dependencies([assert_op]):
        #return K.sum(result, axis=-1)
    #return scale
    return K.sum(result, axis=-1)

def weib_negative_log_like(runtime,scale,shape, tensors = True):
    div = runtime/scale
    
    if tensors:
        return - K.log(shape) + K.log(scale) - (shape-1)*(K.log(runtime)-K.log(scale)) + K.pow(div,shape)
    else:
        return - N.log(shape) + N.log(scale) - (shape-1)*(N.log(runtime)-N.log(scale)) + pow(div,shape)
        

def metric_weib(y_true, y_pred):

    runtime = K.dot(K.constant([1,0,0,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    shapePred = K.dot(K.constant([0,0,1,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    scalePred = K.dot(K.constant([0,0,0,1,0],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    max_nll = K.dot(K.constant([0,0,0,0,1],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    

    scale = K.dot(K.constant([0,1,0,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_pred))
    shape = K.dot(K.constant([0,0,1,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_pred))
 
    scale = scale * (10**8)
    #shape_res = K.abs(shape-shapePred)
    #shape_res = K.log(shape)-K.log(shapePred)
    shape_res = shape - shapePred
    #max_nll = weib_negative_log_like(runtime, scalePred, shapePred)
    pred_nll = weib_negative_log_like(runtime, scale, shape)
    point_max = weib_negative_log_like(runtime, scalePred, shapePred)
    #max_nll = tf.Print(max_nll+1, [max_nll])
    #pred_nll = tf.Print(pred_nll+1, [pred_nll])
    #result = tf.Print(pred_nll - max_nll, [pred_nll - max_nll])
    #result = pred_nll / max_nll
    result = K.abs((pred_nll-point_max)/point_max)
    result = 2*result +(K.abs(shape_res)/shapePred)
    return K.sum(result, axis=-1)

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
 
    scale = scale * 10
 
    #scale = scale * (10**8)
    #loc = loc * (10**6)
    #pun = K.switch(K.less(runtime,loc), loc/runtime, K.zeros_like(loc))
    #location = K.switch(K.less(runtime,loc), runtime - 1, loc)
    # now to calculate the log of the density
 
    percent = K.abs(shape - shapePred)#/shapePred


    res = 0.5 + 0.5*tf.erf((K.log(runtime) - scale)/(N.sqrt(2)*shape))
    res = (empiric - res)
 
    return K.sum(K.abs(res), axis=-1)+K.sum(percent, axis=-1)# + K.sum(K.abs(shape-shapePred), axis = -1)


def ks_par(y_true, y_pred):
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
    #shape = K.switch(K.less(K.abs(shape), eps), zero, shape)
    #shape = K.switch(K.less(shape,eps),K.switch(K.greater(shape,-eps), zero, shape), shape)
    cdfPart = 1+K.switch(K.less(shape, zero), K.switch(K.less(runtime, -scale/shape), (shape*runtime)/scale, zero),(shape*runtime)/scale)
    res = K.switch(K.less(K.abs(shape), eps),1-K.exp(-runtime/scale),1-K.pow(cdfPart, -(1/shape)))
    #res = 1-K.exp(-runtime/scale)
    return K.sum(K.abs(empiric - res), axis = -1)+K.sum(K.abs(shape-shapePred), axis = -1)
    #return K.sum(runtime, axis=-1)

def nll_weib(y_true, y_pred):
    
    
    pun = 0
    K.set_epsilon(1e-12)
    c = K.exp(K.clip(K.abs(y_pred),0,10))*100
    res=0
    punishment = c * K.ones_like(y_pred)
    
    # runtimes contains a 1xBatchSize vector with the considered runtime for the instance
    # assuming the first space of y_true is the runtime
    # similar for loc scale and shape
    
    
    
    # TODO: Check if all clippings make sense
    

    
    runtime = K.clip(K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true)),K.epsilon(),sys.float_info.max)
    #runtime = K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
    #runtime = K.ones(shape = (1,3), dtype = 'float64')
    
    booltensor = K.cast(K.less(y_pred,K.zeros_like(y_pred)),dtype='float64')
    pun = K.sum(K.sum(booltensor * punishment,axis = -1),axis=-1)
    
    
    loc = K.clip(K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred)),K.epsilon(),sys.float_info.max)
    scale = K.clip(K.dot(K.constant([0,1,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred)),K.epsilon(),sys.float_info.max)
    shape = K.clip(K.dot(K.constant([0,0,1],dtype = 'float64',shape = (1,3)),K.transpose(y_pred)),K.epsilon(),sys.float_info.max)
    
    pun = pun + runtime/K.exp(scale)
    
    
    # now to calculate the log of the density
   
    
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
    #shape = K.exp(shape)
    #res = K.log(scale*shape*K.pow(scale*runtime,shape-K.ones_like(shape))*K.exp(-1*K.pow(scale*runtime,shape)))


    res = K.square((K.log(runtime)-scale))/(2*K.square(shapeClip)) +(1/2*math.log(2*math.pi)+K.log(shapeClip)+K.log(runtime))
    res = K.switch(K.less(shape,K.zeros_like(shape)),pun,res)

    return K.sum(res,axis = -1)




def nll_par(y_true, y_pred):
    res=0

    # runtimes contains a 1xBatchSize vector with the considered runtime for the instance
    # assuming the first space of y_true is the runtime
    # similar for loc scale and shape

    runtime = K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true))

    location = K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
    scale = K.dot(K.constant([0,1,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
    shape = K.dot(K.constant([0,0,1],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))


    #booltensor = K.cast(K.less(location,K.zeros_like(location)),dtype='float64')
    #c = K.exp(K.clip(K.abs(location),0,5))
    #punishment = c * K.ones_like(location)
    #pun1 = K.sum(K.sum(booltensor * punishment,axis = -1),axis=-1)

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

    #res = K.log(scale*shape*K.pow(scale*runtime,shape-K.ones_like(shape))*K.exp(-1*K.pow(scale*runtime,shape)))


    #res = -K.log((1/scale)*K.pow((K.ones_like(shape)+(shape*(runtime-loc)/scale)),-(1/shape)-K.ones_like(shape)))
    one = ((1/shape)+K.ones_like(shape))
    two = K.ones_like(shape)+(shape*(runtime-location)/K.exp(scale))
    three = K.log(two)
    res = one * three + scale
    res = K.sum(res,axis = -1) + pun2 + pun3 #+ pun1

    return res
    
