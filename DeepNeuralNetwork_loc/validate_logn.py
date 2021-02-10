from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GaussianNoise
from keras.layers import GaussianDropout
import numpy
#numpy.random.seed(7)
from keras import optimizers
from keras.backend import normalize_batch_in_training
from keras.layers.normalization import BatchNormalization

import functions
import losses.loss_functions_weib as loss
import losses.loss_functions_logn as log_loss
import metrics.metrics_logn as metrics

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
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import os.path

K.set_floatx('float64')

identification=19381
batchSize = 16

k = 10 # k-fold cross validation

mean = [  2.07535642e+03, 8.79820570e+03, 2.02533605e+03, 8.74273727e+03, 2.47030701e-02, 6.34420225e-03, 0.00000000e+00, 2.31650684e-01, 4.95757410e-01, 5.87128817e-01, 0.00000000e+00, 1.00000000e+00, 7.09405974e-01, 1.56175633e-03, 6.32341375e-02, 1.54329748e-03, 2.64027511e-03, 1.55109952e-01, 0.00000000e+00, 0.00000000e+00, 9.65367228e-01, 0.00000000e+00, 1.56175633e-03, 2.88689407e-01, 3.94612841e-04, 3.85964061e-03, 2.71127291e+00, 2.28862633e-01, 1.71263225e-01, 0.00000000e+00, 1.00000000e+00, 3.36639873e+00, 7.66047892e-04, 3.97091176e-01, 1.40171629e-05, 2.19497158e-03, 2.32798653e+00, 4.92875200e-01, 3.14356588e-03, 2.90495192e-01, 7.85261986e-04, 7.88973966e-03, 0.00000000e+00, 2.36417836e-03, 2.44216678e-01, 7.20252509e-04, 5.54463289e-03, 2.96793843e+00, 1.02950014e-01, 2.44245240e-01, 4.32108605e-02, 2.98641242e-01, 2.25805509e+00, 1.11608961e-01, 6.06536507e+01, 2.93483550e-01, 4.44831906e+02, 4.38002016e-02, 4.45092668e+02, 4.25219959e+02, 4.64971487e+02, 7.22220680e-01, 2.21766560e-01, 8.49928395e-01, 6.12104969e-02, 2.22334012e+00, 4.29164764e+01, 2.22621053e-01, 4.45050629e+02, 3.82883515e-02, 4.45125255e+02, 4.25346232e+02, 4.64920570e+02, 1.57544487e-01, 3.72052118e-01, 8.34429868e-01, 2.38244147e-02, 2.59462322e+00, 2.70024754e-01, 4.21481389e-01, 2.00004073e+00]
variance = [  1.60309191e+05, 2.87444998e+06, 1.52709592e+05, 2.83859272e+06, 1.28692080e-05, 1.86540578e-06, 0.00000000e+00, 9.61819332e-07, 1.14980670e-05, 4.04577374e-06, 0.00000000e+00, 0.00000000e+00, 3.61513447e-04, 1.08315440e-07, 3.02106698e-05, 1.05764686e-07, 3.20527678e-07, 3.58884374e-04, 0.00000000e+00, 0.00000000e+00, 2.73196942e-05, 0.00000000e+00, 1.08315440e-07, 3.36773537e-05, 3.22150996e-08, 6.63273369e-07, 3.19596575e-04, 1.27560552e-05, 9.95713427e-06, 0.00000000e+00, 0.00000000e+00, 5.08345694e-04, 2.58375798e-08, 5.49063146e-05, 1.62639343e-09, 2.37222479e-07, 3.07848504e-04, 3.35228739e-05, 4.32562564e-07, 3.75968334e-05, 1.24365306e-07, 2.68286212e-06, 0.00000000e+00, 2.46987898e-07, 2.87581837e-05, 4.14485143e-08, 1.38758085e-06, 4.65266903e-04, 5.43690351e-07, 1.93480803e-05, 9.92144239e-06, 2.21499393e-03, 2.64971174e-04, 6.23684156e-04, 1.37783030e+02, 2.24922362e-02, 7.29165513e+03, 7.07452286e-05, 7.32374447e+03, 6.97745060e+03, 7.64513157e+03, 7.55509818e-04, 6.47058150e-04, 3.55004929e-05, 2.57427884e-02, 1.35584969e-03, 5.30528292e+01, 3.04976832e-02, 7.32298943e+03, 6.91601330e-05, 7.33426181e+03, 7.01028338e+03, 7.64369023e+03, 3.35078570e-04, 2.83971789e-02, 6.43312677e-06, 1.00659625e-03, 3.28351384e-03, 3.92871543e-05, 4.45607588e-04, 8.13004758e-07]

names = []

with open("./chosen_models/names_logn",'r') as f:
    names = f.readlines()
names = list(map(lambda x: x.strip(), names))

feat_label = functions.createSampleData(distribution='lognorm')
feat_label_sorted = list(feat_label)
for elem in feat_label:
    feat_label_sorted[names.index(elem[0])] = elem

# j = 0
# while j < len(feat_label):
#     if feat_label_sorted[j][0] != names[j]:
#         print("not sorted correctly: ", names[j], "at position", j)
#     j += 1

feat_label = feat_label_sorted
del feat_label_sorted

logX, logY = functions.sample(feat_label, k=1/k, seed=identification+1)
valX,validY = functions.createRuntimeData(distribution='lognorm')
distValid = validY
validY = [ [x[0],x[1],x[2], x[3]] for x in validY ]


#metric = [0.0, 0.0, 0.0]
c = 1
#for c in range(k):
## Train network for lognorm

logXtest = list(it.chain(it.chain(* logX[c])))
logYtest = list(it.chain(it.chain(* logY[c])))
logXtrain = list(it.chain(* it.chain(*(logX[:c]+logX[c+1:]))))
logYtrain = list(it.chain(* it.chain(*(logY[:c]+logY[c+1:]))))


#print(weibX)
logXtrain, mean, variance = functions.normalize(logXtrain, mean, variance)
logXtest, mean, variance = functions.normalize(logXtest, mean, variance)
validX, mean, variance = functions.normalize(valX,mean,variance)

distTest = logYtest
logYtest = [ [x[0],x[1],x[2], x[3]] for x in logYtest ]
logYtrain = [ [x[0],x[1],x[2], x[3]] for x in logYtrain ]

filepath = "./chosen_models/log_l_1.2_d_0.32_g_0.05_gh_0.2_layers_2_neurons_32_decay_0.9_id_19381_run_1.h5"

if os.path.isfile(filepath):
    model2 = load_model(filepath,custom_objects={"exp":K.exp,"ks_log":log_loss.ks_log,"rmse_shape_logn":metrics.rmse_shape_logn,"rmse_scale_logn":metrics.rmse_scale_logn})

#print("mean: ", mean, "variance: ", variance)

ev= model2.evaluate(x=validX, y=validY, batch_size=batchSize, verbose=0)
print("validX: ", ev)
ev= model2.evaluate(x=logXtest, y=logYtest, batch_size=batchSize, verbose=0)
print("testData:", ev)
    
    
    
    
    
    
    
    
    #metric = list(map((lambda x,y: x+y), metric, ev))

    # distTest: runtime, i/l, shape, scale, loc, dist

preds = []
maxLs = []

# create list of runtimes to print
runsRaw = [(x[0], x[4]) for x in distTest]

runs = []
app = []
name = ""
for (i, dist) in runsRaw:
    if name == "":
        name = dist
    if name != dist:
        name = dist
        if len(app) != 0:
            while len(app) < 300:
                m = numpy.mean(app)
                app.append(m)
            runs.append(app)

        app = [i]
    else:
        app.append(i)

runs.append(app)


lines = len(runs)

bias = 0
for i in range(0,lines):
    preds.append(model2.predict(numpy.asarray([logXtest[300*i]]),verbose=0))

    maxLs.append([distTest[bias][3],distTest[bias][2]])
    bias = bias + len(runs[i])

s = "model_testData.h5"
# write prediction
with open("./results/predictions/log"+ s, 'w') as f:
    for p in preds:

        tmp = [item for sublist in numpy.asarray(p).tolist() for item in sublist]
        tmp = [tmp[0]*10,tmp[1]]

        f.write(str(tmp).replace("[","").replace("]","").replace(",","") + "\n")

# write maximum Likelihood estimations
with open("./results/estimations/log"+s, 'w') as f:
    for like in maxLs:
        f.write(str(like).replace("[","").replace("]","").replace(",","") + "\n")

# write runtimes
with open("./results/runtimes/log"+s, 'w') as f:
    for r in runs:
        f.write(str(r).replace("[","").replace("]","").replace(",","") + "\n")


#------------------------


preds = []
maxLs = []

# create list of runtimes to print
runsRaw = [(x[0], x[4]) for x in distValid]

runs = []
app = []
name = ""
for (i, dist) in runsRaw:
    if name == "":
        name = dist
    if name != dist:
        name = dist
        if len(app) != 0:
            while len(app) < 300:
                m = numpy.mean(app)
                app.append(m)
            runs.append(app)

        app = [i]
    else:
        app.append(i)

runs.append(app)


lines = len(runs)

bias = 0
for i in range(0,lines):
    preds.append(model2.predict(numpy.asarray([validX[300*i]]),verbose=0))

    maxLs.append([distValid[bias][3],distValid[bias][2]])
    bias = bias + len(runs[i])

s = "model_validData.h5"
# write prediction
with open("./results/predictions/log"+ s, 'w') as f:
    for p in preds:

        tmp = [item for sublist in numpy.asarray(p).tolist() for item in sublist]
        tmp = [tmp[0]*10,tmp[1]]

        f.write(str(tmp).replace("[","").replace("]","").replace(",","") + "\n")

# write maximum Likelihood estimations
with open("./results/estimations/log"+s, 'w') as f:
    for like in maxLs:
        f.write(str(like).replace("[","").replace("]","").replace(",","") + "\n")

# write runtimes
with open("./results/runtimes/log"+s, 'w') as f:
    for r in runs:
        f.write(str(r).replace("[","").replace("]","").replace(",","") + "\n")

#filepath = "./models/log"+"_l_"+str(l)+"_d_"+str(dropout)+"_g_"+str(gauss)+"_gh_"+str(gaussHidden)+"_layers_" + str(hiddenLayers)+ "_neurons_" + str(hiddenLayerNeurons) + "_decay_" + str(lrdecay) +"_id_"+str(identification)+".h5"

#metric = list(map((lambda x: x/k), metric))
#print(filepath, ":", metric)
