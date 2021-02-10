
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

def calc_loc(p):
    alpha = 11.791128437437232
    c=27.419857262609916
    k=0.324040526396282
    return alpha*K.pow(K.pow(1/(1-p), 1/k)-1, 1/c)

def log_square(y_true, y_pred):
    # y_pred = calc_loc(y_pred)
    return K.sqrt(K.mean(K.square(y_true-y_pred)))
