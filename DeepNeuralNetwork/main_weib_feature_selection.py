from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GaussianNoise
from keras.layers import GaussianDropout
import numpy as np

from keras import optimizers
from keras.backend import normalize_batch_in_training
from keras.layers.normalization import BatchNormalization

import functions_feat_sel as functions
import losses.loss_functions_weib_feat_sel as loss
import losses.loss_functions_weib as log_loss
import metrics.metrics_weib_feat_sel as metrics_weib

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
import callback_monitor as expected_monitor



K.set_floatx('float64')

"""
    This program is used to learn a neural network for prediciting the parameters of a Weibull distribution.
    The Weibull distribution describes the runtim behavior of the SAT solver probSAT on a uniformly generated CNF instance.

    Several countermeasures against overfitting are used. Which can be found the arguments below.
    The arguments can then be used for tuning the network.
"""

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dropout", type=float, default=0.01, help="Percentage of the dropout as a float.")
parser.add_argument("-g", "--gauss", type=float, default=0.01, help="Gaussian noise is added to the input. This value describes the variance of the noise.")
parser.add_argument("-gh", "--gaussHidden", type=float, default=0.12, help="Gaussian noise is added to the hidden layers. This value describes the variance of the noise.")
parser.add_argument("-lr", "--lrdecay", type=float, default=0.999639, help="The learning rate is decaying exponentially. This value is the base of the exponential decay.")
parser.add_argument("-hi", "--hidden", type=int, default=8, help="The number of hidden layers. deprecated - not in use anymore.")
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


# start with weibull for testing

hiddenLayers = 2
k = 10 # 10-fold cross validation
feat_label = functions.createSampleData()

# The following functions splits the data into k parts.
runs, weibX, weibY = functions.sample(feat_label, k=1/k, seed=identification+1, dist_names = True)


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
        Split the data in a training part (weibXtrain, weibYtrain) 
        which is activly used for calculating the loss function and adapting the parameters of the networks
        and a test part (weibXtest, weibYtest) used for validating the current performance of the network.

        The variables weibXtrain and weibXtest contain the features.
        The variables weibYtrain and weibYtest contain the 'true' parameters of the distribution.
        The variable test_dists contains runtimes measured on the instances.
    """
    weibXtest = list(it.chain(it.chain(*weibX[c])))
    weibYtest = list(it.chain(it.chain(*weibY[c])))
    weibXtrain = list(it.chain(* it.chain(*(weibX[:c]+weibX[c+1:]))))
    weibYtrain = list(it.chain(* it.chain(*(weibY[:c]+weibY[c+1:]))))
    test_dists = list(it.chain(*runs[c]))

    # The features are normalized such that they have mean 0 and variance 1.
    weibXtrain,foo,bar = functions.normalize(weibXtrain)
    weibXtest, foo, bar = functions.normalize(weibXtest, foo, bar)

    

    dist = weibYtrain
    weibYtrain = [[x[0],x[1],x[2], x[3]] for x in weibYtrain]
    locs = [ x[4] for x in weibYtest]
    weibYtest = [[x[0],x[1],x[2], x[3]] for x in weibYtest]
    """
        The performane of the network is measured by the "expected speedup".
        That is, the parameters of the distribution should be predicted such that they are useful for predicting restart times.
    """ 
    expected_runtime_monitor = expected_monitor.Expected_Runtime_Monitor(weibXtest, test_dists, locs, filepath="./models/id_"+str(identification)+"_c_"+str(c)+".h5", patience = args.patience)


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

     
    model1.add(Dense(4,activation='sigmoid',kernel_regularizer=l2(l)))
     

    adam = optimizers.Adam(lr=learningRate,clipnorm=0.5)
    # A custom loss function is used which is based on the KS test statistic.
    model1.compile(loss=loss.ks_weib,optimizer=adam )



########## MODEL WITH CHECKPOINT ###########
    history = model1.fit(np.array(weibXtrain),np.array(weibYtrain),epochs=maxEpochs, batch_size=batchSize, callbacks=[expected_runtime_monitor,change_lr, terminate], verbose=0)
    metric += expected_runtime_monitor.current_best_metric

print(metric/k)
with open("./results/results.txt", 'a') as f:
    f.write(str(metric/k)+"\n")    
