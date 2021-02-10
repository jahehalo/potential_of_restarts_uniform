from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GaussianNoise
from keras.layers import GaussianDropout
import numpy
#numpy.random.seed(7)
from keras import optimizers
from keras.backend import normalize_batch_in_training
from keras.layers.normalization import BatchNormalization

import functions_feat_sel as functions
import losses.loss_functions_weib_feat_sel as loss

from keras.regularizers import l2#, activity_l2
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
#from tensorflow.errors import InvalidArgumentError
import itertools as it
import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import os.path



K.set_floatx('float64')


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dropout", type=float, default=0.0, help="Percentage of the dropout as a float.")
parser.add_argument("-g", "--gauss", type=float, default=0.0, help="Gaussian noise is added to the input. This value describes the variance of the noise.")
parser.add_argument("-gh", "--gaussHidden", type=float, default=0.0, help="Gaussian noise is added to the hidden layers. This value describes the variance of the noise.")
parser.add_argument("-lr", "--lrdecay", type=float, default=0.5, help="The learning rate is decaying exponentially. This value is the base of the exponential decay.")
parser.add_argument("-hi", "--hidden", type=int, default=8, help="The number of hidden layers.")
parser.add_argument("-l", "--lreg", type=float, default=0.01, help="L2 regularization is used. This value describes the penalty.")

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

#weibX,weibY,logX,logY,parX,parY,weibXtest,weibYtest,logXtest,logYtest,parXtest,parYtest = functions.createRuntimeData()
hiddenLayers = 2
k = 2 # k-fold cross validation
feat_label = functions.createSampleData()
weibX, weibY = functions.sample(feat_label, k=1/k, seed=identification+1)
# weibY = [x[4] for x in weibY]
valX, validY = functions.createRuntimeData()
# validY = [x[4] for x in validY]

#permutation = random.permutation(len(weibXtest))
#weibXtest = [weibXtest[i] for i in permutation]
#weibYtest = [weibYtest[i] for i in permutation]
#print(weibY)



maxEpochs = 3000
batchSize = 16
learningRate = 0.0001
decay = 0.0

def learning_rate_scheduler(epochN):
    lr=0.0005
    border = 0
    if epochN > border:
        lr = lr * lrdecay**(epochN-border)
    return lr

#out1 = normalize_batch_in_training(



#print(weibX)
#weibGenerator()
#quit()

## Train network for weibull


change_lr = callback.LearningRateScheduler(learning_rate_scheduler)
early_stop = EarlyStopping(monitor='val_loss',min_delta=0, patience=50, mode='min')
terminate = TerminateOnNaN()

metric = 0.0
metric_test = 0.0
for c in range(k):
    weibXtest = list(it.chain(it.chain(* weibX[c])))
    weibYtest = list(it.chain(it.chain(* weibY[c])))
    weibXtrain = list(it.chain(* it.chain(*(weibX[:c]+weibX[c+1:]))))
    weibYtrain = list(it.chain(* it.chain(*(weibY[:c]+weibY[c+1:]))))

    #print(weibX)
    weibXtrain,foo,bar = functions.normalize(weibXtrain)
    weibXtest, foo, bar = functions.normalize(weibXtest, foo, bar)
    validX, foo,bar = functions.normalize(valX, foo,bar)

    filepath = "./models/weibull"+"_l_"+str(l)+"_d_"+str(dropout)+"_g_"+str(gauss)+"_gh_"+str(gaussHidden)+"_layers_" + str(hiddenLayers)+ "_neurons_" + str(hiddenLayerNeurons) + "_decay_" + str(lrdecay) +"_id_"+str(identification)+"_run_"+str(c)+".h5"
    if os.path.isfile(filepath):
        os.remove(filepath)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,save_weights_only=False, mode='min')

    model1 = Sequential()


    model1.add(GaussianNoise(gauss, input_shape=(18,)))
    model1.add(Dense(hiddenLayerNeurons, activation='tanh',kernel_regularizer=l2(l)))
    model1.add(BatchNormalization())
    model1.add(GaussianNoise(gaussHidden))
    model1.add(Dropout(dropout))


    
    model1.add(Dense(int(hiddenLayerNeurons/2),activation='tanh',kernel_regularizer=l2(l)))
    model1.add(BatchNormalization())
    model1.add(GaussianNoise(gaussHidden))
    model1.add(Dropout(dropout))

     
    model1.add(Dense(1,activation=K.exp,kernel_regularizer=l2(l)))
     

    adam = optimizers.Adam(lr=learningRate,clipnorm=0.5)
    model1.compile(loss=loss.ks_weib,optimizer=adam)


    history = model1.fit(weibXtrain,weibYtrain,epochs=maxEpochs, validation_data=(weibXtest, weibYtest), batch_size=batchSize, callbacks=[change_lr, checkpoint, early_stop, terminate], verbose=1)

    #print(filepath, ":", history.history)


    pred = "./results/predictions/"+"weib_id_"+str(identification)+"_"+str(c)+"_l_"+str(l) + "_neurons_" + str(hiddenLayerNeurons) + "_decay_" + str(lrdecay) + "_d_"+str(dropout)+"_g_"+str(gauss)+"_gh_"+str(gaussHidden)
    est = "./results/estimations/"+"weib_id_"+str(identification)+"_"+str(c)+"_l_"+str(l) + "_neurons_" + str(hiddenLayerNeurons) + "_decay_" + str(lrdecay) + "_d_"+str(dropout)+"_g_"+str(gauss)+"_gh_"+str(gaussHidden)+"_est"



    if os.path.isfile(filepath):
        model1.load_weights(filepath)

    ev = model1.evaluate(x=weibXtest, y=weibYtest, batch_size=batchSize, verbose=0)
    metric_test = metric_test + ev
    # print("test: ", ev, filepath)
    # p = model1.predict(weibXtest)
    # with open(pred + "_test_pred.txt",'w') as f, open(est + "_test_est.txt", 'w') as g:
    #     i=0
    #     while i < len(weibXtest):
    #         f.write(str(p[i][1]*10**8)+", "+ str(p[i][2])+"\n")
    #         g.write(str(weibYtest[i][3])+", "+str(weibYtest[i][2])+"\n")
    #         i += 300



    ev = model1.evaluate(x=validX, y=validY, batch_size=batchSize, verbose=0)
    print(ev)
    metric = metric + ev
    #print("validation: ", ev,filepath)
    # p = model1.predict(validX)
    # with open(pred + "_val_pred.txt",'w') as f, open(est + "_val_est.txt", 'w') as g:
    #     i=0
    #     while i < len(validX):
    #         f.write(str(p[i][1]*10**8)+", "+ str(p[i][2])+"\n")
    #         g.write(str(validY[i][3])+", "+str(validY[i][2])+"\n")
    #         i += 300

# metric = list(map(lambda c: c/k, metric))
metric_test = metric_test / k
metric = metric / k
print("cross val test:", metric_test,"cross val validation:", metric)
    