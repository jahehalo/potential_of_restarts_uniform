These folders contain supplementary data to "The Potential of Restarts for ProbSAT" on uniform instances.
by Jan-Hendrik Lorenz and Julian Nickerl.
Ulm University 
Institute of Theoretical Computer Science

The original 2400 instances used to train the machine learning components can be found in the "instances" folder.
To train the machine learning components, each instance was solved multiple times to estimate the parameters for the random distributions. The results of these runs can be found in the "outputs" folder.
The corresponding maximum likelihood estimates of the parameters for the lognormal, the Weibull, and the generalized Pareto distributions can then be found in the "dists" folder. In addition, a confidence interval is given for each parameter. Lastly, the corresponding p-values from the KS test are also given for each distribution type.
The folder "dists_eval" is similar, except that it contains only the files for the test set. 

The code in the "variance_threshold" folder was used to select the features that are used for the machine learning components (random forest and the neural networks). The Random Forest was then trained using the code in "randomForests".
The folder "DeepNeuralNetwork" contains the code that was used to train and evaluate the neural network for the Weibulld and the lognormal distributions whereas the folder "DeepNeuralNetwork_loc" contains the code for the location parameter network.

The "NewMLPSolver" contains the improved version of probSAT, where the restart times (on uniform instances) are calculated by the machine learning pipeline.
The "results" folder contains the results from the empirical comparison of the "no restarts" and the "pipeline restarts" strategy. In addition, the results using Luby's strategy can also be found here.
It is also worth mentioning that folder "NewMLPSolver/cnfs/satisfiable" contains the 100 newly generated instances that were used to compare the three strategies.



In order to execute the "pipeline restart" strategy, the featureSAT12 program must also be compiled and placed in the "SAT-features-competition2012" directory. The source code for this program can be downloaded from http://www.cs.ubc.ca/labs/beta/Projects/SATzilla/. In addition, probSAT, which can be found in the directory "NewMLPSolver/probSAT", has to be compiled.

One can then start the pipeline restart strategy by executing main.py. It is expected that the corresponding cnf file is located in the directory "NewMLPSolver/cnfs".
