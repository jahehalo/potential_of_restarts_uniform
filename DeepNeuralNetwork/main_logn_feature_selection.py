from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GaussianNoise
from keras.layers import GaussianDropout
import numpy as np

from keras import optimizers
from keras.backend import normalize_batch_in_training
from keras.layers.normalization import BatchNormalization

import functions_logn_feat_sel as functions
import losses.loss_functions_logn_feat_sel as loss

from keras.regularizers import l2
from keras.regularizers import l1_l2
from keras.regularizers import l1
import keras
import keras.backend as K
import numpy.random as random
import keras.callbacks as callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TerminateOnNaN
import tensorflow as tf

import itertools as it
import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import os.path
import callback_monitor_logn as expected_monitor



K.set_floatx('float64')


"""
    This program is used to learn a neural network for prediciting the parameters of a lognormal distribution.
    The lognormal distribution describes the runtim behavior of the SAT solver probSAT on a uniformly generated CNF instance.

    Several countermeasures against overfitting are used. Which can be found the arguments below.
    The arguments can then be used for tuning the network.
"""

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dropout", type=float, default=0.01, help="Percentage of the dropout as a float.")
parser.add_argument("-g", "--gauss", type=float, default=0.01, help="Gaussian noise is added to the input. This value describes the variance of the noise.")
parser.add_argument("-gh", "--gaussHidden", type=float, default=0.12, help="Gaussian noise is added to the hidden layers. This value describes the variance of the noise.")
parser.add_argument("-lr", "--lrdecay", type=float, default=0.999639, help="The learning rate is decaying exponentially. This value is the base of the exponential decay.")
parser.add_argument("-hi", "--hidden", type=int, default=8, help="The number of hidden layers.")
parser.add_argument("-l", "--lreg", type=float, default=0.01, help="L2 regularization is used. This value describes the penalty.")
parser.add_argument("-p", "--patience", type=int, default=50, help="After patience epochs without an improvement the learning halts.")

parser.add_argument("-s", "--seed", type=int, default=0, help="This seed is used for the initializing the RNG.")
args = parser.parse_args()

hiddenLayerNeurons = args.hidden
lrdecay = args.lrdecay
dropout = args.dropout
gauss = args.gauss
gaussHidden = args.gaussHidden
l = args.lreg
identification = args.seed


# start with lognormal for testing


hiddenLayers = 2
k = 10 # 10-fold cross validation
feat_label = functions.createSampleData()

# The following functions splits the data into k parts.
runs, lognX, lognY = functions.sample(feat_label, k=1/k, seed=identification+1, dist_names = True)




maxEpochs = 1000
batchSize = 16
learningRate = 0.0001
decay = 0.0

# The learning rate is decaying over time. This helps the network to converge.
def learning_rate_scheduler(epochN):
    lr=0.0005
    border = 0
    if epochN > border:
        lr = lr * lrdecay**(epochN-border)
    return lr



change_lr = callback.LearningRateScheduler(learning_rate_scheduler)
terminate = TerminateOnNaN()

metric = 0.0
for c in range(k):
    """ 
        Split the data in a training part (lognXtrain, lognYtrain) 
        which is activly used for calculating the loss function and adapting the parameters of the networks
        and a test part (lognXtest, lognYtest) used for validating the current performance of the network.

        The variables lognXtrain and lognXtest contain the features.
        The variables lognYtrain and lognYtest contain the 'true' parameters of the distribution.
        The variable test_dists contains runtimes measured on the instances.
    """
    lognXtest = list(it.chain(it.chain(*lognX[c])))
    lognYtest = list(it.chain(it.chain(*lognY[c])))
    lognXtrain = list(it.chain(* it.chain(*(lognX[:c]+lognX[c+1:]))))
    lognYtrain = list(it.chain(* it.chain(*(lognY[:c]+lognY[c+1:]))))
    test_dists = list(it.chain(*runs[c]))

    # The features are normalized such that they have mean 0 and variance 1.
    lognXtrain,foo,bar = functions.normalize(lognXtrain)
    lognXtest, foo, bar = functions.normalize(lognXtest, foo, bar)

    dist = lognYtrain
    lognYtrain = [[x[0],x[1],x[2]] for x in lognYtrain]
    lognYtest = [[x[0],x[1],x[2]] for x in lognYtest]
    """
        The performane of the network is measured by the "expected speedup".
        That is, the parameters of the distribution should be predicted such that they are useful for predicting restart times.
    """ 
    expected_runtime_monitor = expected_monitor.Expected_Runtime_Monitor(lognXtest, test_dists, filepath="./models/id_"+str(identification)+"_c_"+str(c)+".h5", patience = args.patience)

    """
        The network uses 34 features and has two hidden layers
        with 14 and 7 neurons, respectively.
    """

    model1 = Sequential()

    model1.add(GaussianNoise(gauss, input_shape=(34,)))
    model1.add(Dense(14, activation='tanh',kernel_regularizer=l2(l)))
    model1.add(BatchNormalization())
    model1.add(GaussianNoise(gaussHidden))
    model1.add(Dropout(dropout))


    model1.add(Dense(7,activation='tanh',kernel_regularizer=l2(l)))
    model1.add(BatchNormalization())
    model1.add(GaussianNoise(gaussHidden))
    model1.add(Dropout(dropout))

    model1.add(Dense(3,activation='sigmoid',kernel_regularizer=l2(l)))
     

    adam = optimizers.Adam(lr=learningRate,clipnorm=0.5)
    model1.compile(loss=loss.ks_log,optimizer=adam )



########## MODEL WITH CHECKPOINT ###########
    history = model1.fit(np.array(lognXtrain),np.array(lognYtrain),epochs=maxEpochs, batch_size=batchSize, callbacks=[expected_runtime_monitor,change_lr, terminate], verbose=0)
    metric += expected_runtime_monitor.current_best_metric

print(metric/k)
with open("./results/results_logn.txt", 'a') as f:
    f.write(str(metric/k)+"\n")    
