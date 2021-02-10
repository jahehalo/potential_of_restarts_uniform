
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
    """ 
        The neural network produces a number Y between 0 and 1. 
        This function convertes this number Y to the associated scale parameter.
    """
    mu=  2.686333243803358  
    sigma= 0.090459464555333
    quantile = special_math.ndtri(p)/N.sqrt(2.0)
    return K.exp(mu+sigma*N.sqrt(2.0)*quantile)

def shape_calc(p):
    """ 
        The neural network produces a number Y between 0 and 1. 
        This function convertes this number Y to the associated shape parameter.
    """
    alpha = 0.912047580559537 
    c=10.966647985287899 
    k=0.705838374748621 
    return alpha*K.pow(K.pow(1/(1-p), 1/k)-1, 1/c)


def ks_log(y_true, y_pred):
    """
        This function calculates the loss function for the lognormal distribution.
        The loss is based on the KS test statistic.
    
        runtimes contains a 1xBatchSize vector with the considered runtime for the instance
        assuming the first space of y_true is the runtime
        similar for loc scale and shape 
    """

    # The labels contain a measured runtime (in flips), 
    # the associated value of the ecdf and the 'true' shape parameter.
    runtime = K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
    empiric = K.dot(K.constant([0,1,0],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
    shapePred = K.dot(K.constant([0,0,1],dtype = 'float64',shape = (1,3)),K.transpose(y_true))
 
    # The NN predicts a scale and a shape parameter for the lognormal distribution.
    scale = K.dot(K.constant([1,0,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
    shape = K.dot(K.constant([0,1,0],dtype = 'float64',shape = (1,3)),K.transpose(y_pred))
 
    # However, the outputs are scaled to (0,1). 
    # The following steps reverse this scaling to obtain the true estimates of the scale and shape.
    scale = scale_calc(scale)
    shape = shape_calc(shape)
 
    # This part of the loss function ensures that the predicted shape approaches the 'true' shape.
    percent = K.abs(shape - shapePred)#/shapePred

    # The next line calculates the value of the cdf of a lognormal distribution.
    res = 0.5 + 0.5*tf.erf((K.log(runtime) - scale)/(N.sqrt(2)*shape))
    # By subtracting the ecdf, we obtain a value similar to the KS test statistic.
    res = (empiric - res)
 
    return K.sum(K.abs(res), axis=-1)+K.sum(percent, axis=-1)

