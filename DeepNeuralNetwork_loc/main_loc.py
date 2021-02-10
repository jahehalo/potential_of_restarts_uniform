from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GaussianNoise
from keras.layers import GaussianDropout
import numpy

from keras import optimizers
from keras.backend import normalize_batch_in_training
from keras.layers.normalization import BatchNormalization

import functions_feat_sel as functions
import losses.loss_functions_weib_feat_sel as loss

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



K.set_floatx('float64')

"""
    This program is used to learn a neural network for prediciting the location parameter of a Weibull distribution.
    The Weibull distribution describes the runtim behavior of the SAT solver probSAT on a uniformly generated CNF instance.

    Several countermeasures against overfitting are used. Which can be found the arguments below.
    The arguments can then be used for tuning the network.
"""

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dropout", type=float, default=0.01, help="Percentage of the dropout as a float.")
parser.add_argument("-g", "--gauss", type=float, default=0.01, help="Gaussian noise is added to the input. This value describes the variance of the noise.")
parser.add_argument("-gh", "--gaussHidden", type=float, default=0.12, help="Gaussian noise is added to the hidden layers. This value describes the variance of the noise.")
parser.add_argument("-lr", "--lrdecay", type=float, default=0.999639, help="The learning rate is decaying exponentially. This value is the base of the exponential decay.")
parser.add_argument("-hi1", "--hidden1", type=int, default=14, help="The number of neurons in the first hidden layer.")
parser.add_argument("-hi2", "--hidden2", type=int, default=1, help="The number of neurons in the second hidden layer.")
parser.add_argument("-l", "--lreg", type=float, default=0.01, help="L2 regularization is used. This value describes the penalty.")

parser.add_argument("-s", "--seed", type=int, default=0, help="This seed is used for the initializing the RNG.")
args = parser.parse_args()

hiddenLayerNeurons = 1
lrdecay = args.lrdecay
dropout = args.dropout
gauss = args.gauss
gaussHidden = args.gaussHidden
l = args.lreg
identification = args.seed


hiddenLayers = 2
k = 10 # k-fold cross validation
feat_label = functions.createSampleData()

# The following functions splits the data into k parts.
weibX, weibY = functions.sample(feat_label, k=1/k, seed=identification+1)
valX, validY = functions.createRuntimeData()


maxEpochs = 3000
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


## Train network for weibull


change_lr = callback.LearningRateScheduler(learning_rate_scheduler)
early_stop = EarlyStopping(monitor='val_loss',min_delta=0, patience=100, mode='min')
terminate = TerminateOnNaN()

metric = 0.0
metric_test = 0.0
for c in range(k):
    """ 
        Split the data in a training part (weibXtrain, weibYtrain) 
        which is activly used for calculating the loss function and adapting the parameters of the networks
        and a test part (weibXtest, weibYtest) used for validating the current performance of the network.

        The variables weibXtrain and weibXtest contain the features.
        The variables weibYtrain and weibYtest contain the 'true' parameters of the distribution.
    """

    weibXtest = list(it.chain(it.chain(* weibX[c])))
    weibYtest = list(it.chain(it.chain(* weibY[c])))
    weibXtrain = list(it.chain(* it.chain(*(weibX[:c]+weibX[c+1:]))))
    weibYtrain = list(it.chain(* it.chain(*(weibY[:c]+weibY[c+1:]))))

    # The features are normalized such that they have mean 0 and variance 1.
    weibXtrain,foo,bar = functions.normalize(weibXtrain)
    weibXtest, foo, bar = functions.normalize(weibXtest, foo, bar)
    validX, foo,bar = functions.normalize(valX, foo,bar)

    filepath = "./models/loc_id_"+str(identification)+"_c_"+str(c)+"_features_34_no_scaleing"+".h5"
    if os.path.isfile(filepath):
        os.remove(filepath)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,save_weights_only=False, mode='min')


    """
        The network uses 34 features and has two hidden layers
        with 14 and 1 neurons, respectively.
    """


    model1 = Sequential()


    model1.add(GaussianNoise(gauss, input_shape=(34,)))
    model1.add(Dense(args.hidden1, activation='tanh',kernel_regularizer=l2(l)))
    model1.add(BatchNormalization())
    model1.add(GaussianNoise(gaussHidden))
    model1.add(Dropout(dropout))


    
    model1.add(Dense(args.hidden2,activation='tanh',kernel_regularizer=l2(l)))
    model1.add(BatchNormalization())
    model1.add(GaussianNoise(gaussHidden))
    model1.add(Dropout(dropout))

     
    model1.add(Dense(1,activation=K.exp,kernel_regularizer=l2(l)))
     

    adam = optimizers.Adam(lr=learningRate)
    model1.compile(loss=loss.log_square,optimizer=adam)


    history = model1.fit(numpy.array(weibXtrain),numpy.array(weibYtrain),epochs=maxEpochs, validation_data=(numpy.array(weibXtest), numpy.array(weibYtest)), batch_size=batchSize, callbacks=[change_lr, checkpoint, early_stop, terminate], verbose=1)



    if os.path.isfile(filepath):
        model1.load_weights(filepath)

    ev = model1.evaluate(x=numpy.array(weibXtest), y=numpy.array(weibYtest), batch_size=batchSize, verbose=0)
    metric_test = metric_test + ev
    
metric_test = metric_test / k
metric = metric / k
print("cross val test:", metric_test,"cross val validation:", metric, filepath)
    