from sklearn.ensemble import RandomForestClassifier
import sklearn.base
import functions as f
import sys
import getopt
import numpy.random as rand
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.metrics import recall_score
import numpy as np

# This list is later used to output which features are actually used.
feature_names_orig = [['nvarsOrig', 'nclausesOrig', 'nvars', 'nclauses', 'reducedVars', 'reducedClauses', 
    'Pre-featuretime', 'vars-clauses-ratio', 'POSNEG-RATIO-CLAUSE-mean', 'POSNEG-RATIO-CLAUSE-coeff-variation', 
    'POSNEG-RATIO-CLAUSE-min', 'POSNEG-RATIO-CLAUSE-max', 'POSNEG-RATIO-CLAUSE-entropy', 'VCG-CLAUSE-mean', 
    'VCG-CLAUSE-coeff-variation', 'VCG-CLAUSE-min', 'VCG-CLAUSE-max', 'VCG-CLAUSE-entropy', 'UNARY', 'BINARY+', 
    'TRINARY+', 'Basic-featuretime', 'VCG-VAR-mean', 'VCG-VAR-coeff-variation', 'VCG-VAR-min', 'VCG-VAR-max', 
    'VCG-VAR-entropy', 'POSNEG-RATIO-VAR-mean', 'POSNEG-RATIO-VAR-stdev', 'POSNEG-RATIO-VAR-min', 'POSNEG-RATIO-VAR-max', 
    'POSNEG-RATIO-VAR-entropy', 'HORNY-VAR-mean', 'HORNY-VAR-coeff-variation', 'HORNY-VAR-min', 'HORNY-VAR-max', 'HORNY-VAR-entropy', 
    'horn-clauses-fraction', 'VG-mean', 'VG-coeff-variation', 'VG-min', 'VG-max', 'KLB-featuretime', 'CG-mean', 'CG-coeff-variation', 
    'CG-min', 'CG-max', 'CG-entropy', 'cluster-coeff-mean', 'cluster-coeff-coeff-variation', 'cluster-coeff-min', 'cluster-coeff-max', 
    'cluster-coeff-entropy', 'CG-featuretime', 'saps_BestSolution_Mean', 'saps_BestSolution_CoeffVariance', 
    'saps_FirstLocalMinStep_Mean', 'saps_FirstLocalMinStep_CoeffVariance', 'saps_FirstLocalMinStep_Median', 
    'saps_FirstLocalMinStep_Q.10', 'saps_FirstLocalMinStep_Q.90', 'saps_BestAvgImprovement_Mean', 
    'saps_BestAvgImprovement_CoeffVariance', 'saps_FirstLocalMinRatio_Mean', 'saps_FirstLocalMinRatio_CoeffVariance', 
    'ls-saps-featuretime', 'gsat_BestSolution_Mean', 'gsat_BestSolution_CoeffVariance', 'gsat_FirstLocalMinStep_Mean', 
    'gsat_FirstLocalMinStep_CoeffVariance', 'gsat_FirstLocalMinStep_Median', 'gsat_FirstLocalMinStep_Q.10', 'gsat_FirstLocalMinStep_Q.90', 
    'gsat_BestAvgImprovement_Mean', 'gsat_BestAvgImprovement_CoeffVariance', 'gsat_FirstLocalMinRatio_Mean', 
    'gsat_FirstLocalMinRatio_CoeffVariance', 'ls-gsat-featuretime', 'lobjois-mean-depth-over-vars', 'lobjois-log-num-nodes-over-vars', 
    'lobjois-featuretime'
    ]]

inFolder = "./results"
featFolder = "./crossVal/feats"

seed = 1

rand.seed(seed)

features,labels = f.createData(inFolder, featFolder)

# This list is used to obtain the indices of the used features.
feature_indices = [np.asarray(range(81))]

# First, remove all features with variance zero.
# This is because the min_max scaling does not work for those.
sel = VarianceThreshold(threshold=(0))
features = sel.fit_transform(features)
feature_indices = sel.transform(feature_indices)
feature_names = sel.transform(feature_names_orig)

# Then, normalize all features such that they have minimum 0 and maximum 1.
min_max_scaler = preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(features)
print(features[1])
print(len(features[1]))

# Next, remove all features having a variance of less than 0.05 (after normalizing.)
sel = VarianceThreshold(threshold=(5e-2))
sel.fit_transform(features)

# Print the names and indices of the remaining features.
feature_indices = sel.transform(feature_indices)
feature_names = sel.transform(feature_names)
print(feature_indices)
print(feature_names)
print(len(sel.get_support(True)))

# Output the names of the handpicked features.
#The original indices of the handpicked features are: [25, 41, 46, 55]
print("handpicked features:")
print(f"{feature_names_orig[0][25]}, {feature_names_orig[0][41]}, {feature_names_orig[0][46]}, {feature_names_orig[0][55]},")