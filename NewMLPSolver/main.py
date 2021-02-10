import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import functions as f
import sklearn.linear_model as lm
from sklearn.ensemble import RandomForestClassifier
import subprocess
import sys
import math
import random
from keras.models import load_model
import imports.metrics.metrics_logn as metrics_logn
import imports.metrics.metrics_weib as metrics_weib
import imports.metrics.metrics_par as metrics_par
import imports.losses.loss_functions_logn as loss_logn
import imports.losses.loss_functions_weib as loss_weib
import imports.losses.loss_functions_par as loss_par
import keras.backend as K
import numpy as N

import losses.loss_functions_weib_feat_sel as loss_weib
import losses.loss_functions_logn_feat_sel as loss_logn
import losses.loss_loc as loss_loc

"""
    This program first predicts the distribution type (lognormal or Weibull),
    then calculates an optimal restart time,
    then calls probSAT with the appropriate restart strategy
    for a given instance.

"""

K.set_floatx('float64')

writeToFile = False
debug = False
external = True

# Setup the paths to the random forest.
forestPath = "./randomforest/"
cnfPath = ""
seed = 1
featurePath = ""

if writeToFile:
    cnfPath = sys.argv[1]    
    featurePath = sys.argv[2]

if external:
    cnfPath = sys.argv[1]
    seed = 1 if abs(int(sys.argv[2])) == 0 else abs(int(sys.argv[2]))
    featurePath = cnfPath.replace("cnfs", "features") + ".feat"
else:
    print("Please provide a path to a CNF file.")
    quit()
    

if debug:
    print('++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Working on file ' + cnfPath.split("/").pop())


    print('Loading Random Forest ...',end="       ")
randomForest = f.loadClf(forestPath + 'classifier.clf')

if debug:
    print('DONE')
#
    
    print('Loading Neural Networks ...',end="     ")


"""
    The following values are used to normalize the features.
    After normalization they have (approx.) mean 0 and variance 1.
""" 
customLocObj = {'exp':K.exp,'log_square':loss_loc.log_square}
mean_loc = [2.01322751e+03,8.53979592e+03,1.96528345e+03,8.48671277e+03,1.61369548e-03,1.59467154e-03,2.71623053e-03,1.61369548e-03,4.17116500e-04,3.95896280e-03,7.92634724e-04,1.73729055e-05,3.24656433e-03,8.29264035e-04,8.07015967e-03,2.44420606e-03,5.70864137e-03,1.07664399e-01,6.10448082e+01,2.79579961e-01,4.31453423e+02,4.37737859e-02,4.31677627e+02,4.12173847e+02,4.51229025e+02,7.29824450e-01,4.31321550e+01,4.31671995e+02,3.89030666e-02,4.31718821e+02,4.12263039e+02,4.51236584e+02,1.60051212e-01,2.70274639e-01]
var_loc = [1.68570309e+05,3.03152085e+06,1.60824215e+05,2.99597732e+06,1.17721510e-07,1.15298993e-07,3.45780505e-07,1.17721510e-07,3.63557653e-08,7.53286752e-07,2.84772838e-08,2.06544810e-09,4.69911956e-07,1.41153403e-07,3.03048336e-06,2.69613368e-07,1.55113845e-06,6.39367342e-04,1.50604613e+02,1.96613531e-02,7.69035450e+03,6.65551428e-05,7.71764948e+03,7.35453969e+03,8.06238217e+03,8.03571673e-04,5.80710814e+01,7.71363595e+03,6.83944290e-05,7.72452411e+03,7.37987941e+03,8.05854645e+03,3.59693886e-04,4.10233282e-05]

customWeibObj = {'ks_weib':loss_weib.ks_weib}
mean_weib = [1954.88211, 8296.33956, 1908.75669, 8245.24844, 0.00166137525, 0.00164190559, 0.0027927715, 0.00166137525, 0.000438338333, 0.00404224835, 0.000816380225, 2.01835012e-05, 0.00334105653, 0.000871057624, 0.00822667858, 0.00251664546, 0.00585817864, 0.104071866, 60.6163902, 0.271220443, 419.000882, 0.0438812485, 419.225813, 400.002311, 438.451767, 0.735611736, 42.8525355, 419.191489, 0.039366385, 419.204526, 400.025403, 438.512502, 0.162462602, 0.270697071]
var_weib = [166797.721, 3006463.9, 159142.706, 2971140.7, 1.19579572e-07, 1.17475936e-07, 3.57662403e-07, 1.19579572e-07, 3.81476088e-08, 7.74858593e-07, 2.90240793e-08, 2.46783843e-09, 4.76845087e-07, 1.48195687e-07, 3.12648714e-06, 2.73743554e-07, 1.63035859e-06, 0.000611310343, 155.724786, 0.0171712072, 7609.73407, 6.13029899e-05, 7637.15745, 7273.98016, 7980.01829, 0.000774120693, 59.8077927, 7629.06465, 6.35640695e-05, 7638.85063, 7292.6231, 7978.94374, 0.000365246039, 4.1233966e-05]

customLognObj = {'ks_log':loss_logn.ks_log}
mean_logn = [2080.43876, 8819.7532, 2030.38757, 8764.3181, 0.00155840192, 0.00154003222, 0.00263461157, 0.00155840192, 0.000393985119, 0.00384799711, 0.000764750536, 1.43950658e-05, 0.00313674446, 0.000784140835, 0.00786477904, 0.00235937177, 0.00552175249, 0.111992687, 61.1071675, 0.287180172, 445.920878, 0.043281957, 446.149909, 426.288848, 466.053016, 0.72267742, 43.191708, 446.088428, 0.0382539351, 446.146252, 426.411335, 465.983547, 0.157319124, 0.269867898]
var_logn = [161719.734, 2899675.54, 154182.961, 2864966.04, 1.09282222e-07, 1.06657455e-07, 3.12910171e-07, 1.09282222e-07, 3.32795541e-08, 6.75464829e-07, 2.60914725e-08, 1.63599476e-09, 4.36682882e-07, 1.2937754e-07, 2.73377222e-06, 2.49131032e-07, 1.38259317e-06, 0.000634237606, 140.171495, 0.0218341868, 7365.13667, 7.27735525e-05, 7395.32122, 7046.61858, 7717.42863, 0.00076193672, 54.0451506, 7387.02692, 7.0620667e-05, 7398.19159, 7075.48528, 7713.34342, 0.000339388238, 3.99739044e-05]


# Load the final models for the Weibull and the lognormal distribution
# as well as the model for the location parameter.
weibNet = load_model("./networks/weibull.h5",custom_objects = customWeibObj)
lognNet = load_model("./networks/lognorm.h5",custom_objects = customLognObj)
locNet = load_model("./networks/location.h5",custom_objects = customLocObj)
    
    
if debug:    
    print('DONE')



iterations = 1

if debug:
    print('Calculating Features ...',end="        ")
featureVec = []

for i in range(iterations):
    if debug:
        print("progress: " + str(i+1) + " of " + str(iterations))

    features = []

    if featurePath != "": 
        # This case is executed if the features are precomputed and provided in a file.
        with open(featurePath,'r') as fl:
            """
                Previously, some features have been filtered such that they have a minimum degree of variance.
                The filtering was performed by first scaling them to have minimum 0 and maximum 1.
                Features with a minimum variance of 0.05 have been taken.
                Four additional features have also been selected. The handpicked features are [25, 41, 46, 55].
                All other features are filtered out and not considered anymore.
            """
            fl.readline()
            features = list(map(float,fl.readline().split(',')))
            features = [features[i] for i in [ 0, 1, 2, 3, 13, 15, 16, 22, 24, 25, 32, 34, 38, 40, 41, 43, 46, 53, 54, 55, 56, 57, 58, 59, 60, 61, 66, 68, 69, 70, 71, 72, 73, 78]]
    else:
        # This case is executed if the features are not precomputed.
        featuresOutput = subprocess.check_output('../SAT-features-competition2012/featuresSAT12 -base -ls -lobjois ' + cnfPath, shell=True) 
        """
            Previously, some features have been filtered such that they have a minimum degree of variance.
            The filtering was performed by first scaling them to have minimum 0 and maximum 1.
            Features with a minimum variance of 0.05 have been taken.
            Four additional features have also been selected.
            All other features are filtered out and not considered anymore.
        """
        featuresOutput = str(featuresOutput).split('\\n')
        features = list(map(float,featuresOutput[len(featuresOutput)-2].split(',')))
        features = [features[i] for i in [ 0, 1, 2, 3, 13, 15, 16, 22, 24, 25, 32, 34, 38, 40, 41, 43, 46, 53, 54, 55, 56, 57, 58, 59, 60, 61, 66, 68, 69, 70, 71, 72, 73, 78]]
        
    featureVec.append(features)
if debug:
    print('DONE')
    print('Predictions:')

# Predict the distribution type.
# Only lognormal and Weibull distributions are possible results.
dist = randomForest.predict([featureVec[0]])

location = 0
a = 0
b = 0
mu = 0
sigma = 0

t = 'luby'
dist = dist[0]
if debug:
    print("  -  dist:", dist)


start = 0
shapeStart = 0
scaleStart = 0
end = 0
shapeEnd = 0
scaleEnd = 0

shapeLow = 0
scaleLow = 0
    
    

if dist == 'none':
    # This case should not occur, it is only used for completeness sake.
    t = 'luby'
    
      
elif dist == 'weibull':
    """
        The random forest predicted the Weibull distribution as the most fitting model.
        First, normalize the features for the Weibull and the location network.
        Then, obtain the predictions of the parameters.
        Lastly, use the parameters to calculate the restart time.
    """
    Data_weib = [N.asarray([0 if d == 0 else c/d for c,d in zip([a-b for a,b in zip(features,mean_weib)],N.sqrt(var_weib))]).reshape((1,34))]    
    Data_loc = [N.asarray([0 if d == 0 else c/d for c,d in zip([a-b for a,b in zip(features,mean_loc)],N.sqrt(var_loc))]).reshape((1,34))] 
    res = weibNet.predict(Data_weib)
    a = f.scale_calc_weib(res[0][1])
    b = f.shape_calc_weib(res[0][2])
    
    res = locNet.predict(Data_loc)
    loc = N.exp(res[0][0])
    
    if debug:
        print("  -  loc:", loc)
        print("  -  a:", a)  # scale
        print("  -  b:", b)  # shape
    
    t = f.weibullRestartTime(a,b,loc)
    
    if debug:
        print("  -  Optimal restart time:", t)
    
    
elif dist == 'lognorm':
    """
        The random forest predicted the lognormal distribution as the most fitting model.
        First, normalize the features for the lognormal network.
        Then, obtain the predictions of the parameters.
        Lastly, use the parameters to calculate the restart time.
    """
    Data = [N.asarray([0 if d == 0 else c/d for c,d in zip([a-b for a,b in zip(features,mean_logn)],N.sqrt(var_logn))]).reshape((1,34))]
    res = lognNet.predict(Data)
    
    mu = f.scale_calc_logn(res[0][0])
    sigma = f.shape_calc_logn(res[0][1])
    
    if debug:
        print("  -  mu:", mu)  # scale
        print("  -  sigma:", sigma)  # shape
    
    t = f.lognormRestartTime(mu,sigma)
    
    
    if debug:
        print("  -  Optimal restart time:", t)   
    


if debug:
    print('Start Solving ...',end="               ")
    
""" 
    Here, probSAT is called with the predicted restart strategy.
    If the Weibull distribution was used and a shape parameter greater than 1.0 was predicted,
    then no restarts are performed.

    If the lognormal distribution was used and a shape parameter less than 0.7 was predicted,
    then no restarts are performed because they do not have a practically big impact.

    In all other cases, a fixed-cutoff strategy is used.
"""
result = f.startSolving(cnfPath, t, seed, debug, int(features[0]))

if debug:
    print('DONE')

if debug:
    print('result: ', result)
else:
    print(result + " " + str(seed))

if writeToFile:

    with open("./predictions/" + cnfPath.split('/').pop().replace('.cnf','.pred'),'w') as fl:
        fl.write("dist " + str(dist[0]) + "\n")
        
        if dist == 'weibull':
            fl.write("loc " + str(loc) + "\n")
            fl.write("a " + str(a) + "\n")
            fl.write("b " + str(b) + "\n")
            fl.write("t " + str(t) + "\n")
            
        elif dist == 'lognorm':
            fl.write("mu " + str(mu) + "\n")
            fl.write("sigma " + str(sigma) + "\n")
            fl.write("t " + str(t) + "\n")
            
        elif dist == 'pareto':
            fl.write("loc " + str(loc) + "\n")
            fl.write("shape " + str(shape) + "\n")
            fl.write("scale " + str(scale) + "\n")
            fl.write("t " + str(t) + "\n")
if debug:
    print('++++++++++++++++++++++++++++++++++++++++++++++++++')
