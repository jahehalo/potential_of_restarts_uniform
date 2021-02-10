
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

# order of parameters: (loc),scale,shape
  
def ks_weib(y_true, y_pred):
    
    # runtimes contains a 1xBatchSize vector with the considered runtime for the instance
    # assuming the first space of y_true is the runtime
    # similar for loc scale and shape
    
    
    
    # TODO: Check if all clippings make sense
    
    
    runtime = K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
    #runtime = K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
    #runtime = K.ones(shape = (1,3), dtype = 'float64')
    empiric = K.dot(K.constant([0,1,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
    shapePred = K.dot(K.constant([0,0,1],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
    
    #loc =   K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
    scale = K.dot(K.constant([0,1,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
    shape = K.dot(K.constant([0,0,1],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
    

    scale = scale * (10**8)
    #loc = loc * (10**6)
    #pun = K.switch(K.less(runtime,loc), loc/runtime, K.zeros_like(loc))
    #location = K.switch(K.less(runtime,loc), runtime - 1, loc)
    # now to calculate the log of the density
   
    
    #res = K.switch(K.less(runtime,loc),K.zeros_like(loc),1-K.exp(-K.pow((runtime-location)/scale,shape)))
    res = 1-K.exp(-K.pow(runtime/scale,shape))
    res = (empiric - res)
    
    return K.sum(K.abs(res), axis=-1)+K.sum(K.abs(shape-shapePred), axis = -1)


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

     
    res = 0.5 + 0.5*tf.erf((K.log(runtime) - scale)/(numpy.sqrt(2)*shape))
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
    
