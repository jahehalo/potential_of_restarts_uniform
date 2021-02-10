from keras.callbacks import Callback
import numpy as np
from scipy.special import gamma
from scipy.special import gammainc
from scipy.stats import norm
from scipy.special import erf
from scipy.special import erfinv
import copy

class Expected_Runtime_Monitor(Callback):

    # X_test are the features
    # each run should be a list of runtimes
    def __init__(self, X_test, runs, filepath = "./model.h5", patience = 50):
        observed_runs = set()
        self.timerReset = patience
        self.X_test = []
        self.runs = []
        self.locations = []
        for i in range(len(runs)):
            l = str(runs[i])
            if l in observed_runs:
                continue

            observed_runs.add(l)
            self.X_test.append(X_test[i])
            self.runs.append(np.array(runs[i]))
        self.X_test = np.array(self.X_test)
        self.runs = np.array(self.runs)
        self.filepath = filepath



    def scale_calc(self, p):
        """ 
            The neural network produces a number Y between 0 and 1. 
            This function convertes this number Y to the associated scale parameter.
        """ 
        mu=   2.686333243803358
        sigma=   0.090459464555333 
        return np.exp(mu + sigma*norm.ppf(p))

    def shape_calc(self, p):
        """ 
            The neural network produces a number Y between 0 and 1. 
            This function convertes this number Y to the associated shape parameter.
        """ 
        alpha= 0.912047580559537
        c=  10.966647985287899
        k=   0.705838374748621 
        return alpha*np.power(np.power(1/(1-p), 1/k)-1, 1/c)

    def restart_time(self,predictions):
        """
            This function calculates restart times for the associated lognormal distributions.
        """
        scales = [self.scale_calc(x[0]) for x in predictions]
        shape = [self.shape_calc(x[1]) for x in predictions]
        restarts = []

        for i in range(len(scales)):
            restarts.append(self.lognormRestartTime(scales[i], shape[i]))

        return restarts

    def on_train_begin(self, logs={}):
        """
            Setup the monitor. The currently best model is saved.
        """
        self.speedups = []
        self.current_best_metric = 0.0
        self.current_best_model = None
        self.timeout = self.timerReset
        
        
    def on_train_end(self, logs={}):
        """
            After the end of training the best model is saved in a file for later analysis.
            Furthermore, the final predicted restart times are also stored.
        """
        predictions = self.model.predict(self.X_test)
        predictions = self.restart_time(np.array(predictions))
        print("training ended")                     
        self.current_best_model.save(self.filepath, overwrite=True)        
                

    def on_epoch_end(self, batch, logs={}):
        """
            After each epoch, the current empirical speedup is calculated.
            The geometric mean is used for the average empirical speedup.
        """

        # The restart times are calculated based on the current predictions of the NN.
        predictions = self.model.predict(self.X_test)
        predictions = self.restart_time(predictions)
        speedup = []
        for i in range(len(predictions)):
            # calculate_mean computes the empirical speedup when using the predicted restart time.
            # First applying the logarithm to those means...
            speedup.append(np.log(self.calculate_mean(predictions[i], self.runs[i])))
        
        # ... and afterwards the exp-functions yields the geometric mean.
        metric = np.exp(np.mean(speedup))
        self.speedups.append(metric)
        # Lastly, check whether the current model is an improvement according to the average speedup.
        if metric > self.current_best_metric:
            self.current_best_metric = metric
            self.timeout = self.timerReset
            self.current_best_model = copy.copy(self.model)
        else:
            self.timeout -= 1
        print("  ",np.exp(np.mean(speedup)))

        if self.timeout == 0:
            self.model.stop_training = True
            

    def calculate_mean(self, restart, observed):
        """
            This function calculates the speedup factor.
            The argument restart contains the restart time.
            The argument observed contains a list of observed runtimes.
        """
        if np.isinf(restart):
            # an infinite restart time corresponds to no restarts.
            # Then, there is neither a speedup nor a slowdown.
            return 1.0 
        l = np.searchsorted(observed, restart) 
        conditional_runtimes = observed[:l]  
        if len(conditional_runtimes) == 0:
            conditional_runtimes = [restart]
        p = l/len(observed)
        if l == 0:
            p += 1/len(observed)*np.exp((restart-observed[0])/(observed[0])*1e+1)
        elif p < 1.0:
            p += 1/len(observed) * (restart-observed[l-1])/(observed[l]-observed[l-1])
        result = (1-p)/p * restart + np.mean(conditional_runtimes)

        if restart < observed[0]:
            result = result * observed[0]/restart

        if np.isnan(p) or p == 0.0 or result == 0:
            print(restart, observed, conditional_runtimes)

        return np.mean(observed)/result


    def lognormRestartTime(self,muIn,sigma,maxIterations = 100, errorBoundary = 1.0e-10):
        """
            This functions computes the optimal restart time of a lognormal distribution with 
            scale muIn and shape sigma.
        """
        
    
        if sigma < 0.7: # at this point, restarts have been theoretically useful for quite some time, but will most likely never have any relevance in practice
            return np.inf   
    
        # function q is the quantile function, r is its derivative and s is the anti-derivative.
        q = lambda p,mu: np.exp(mu + sigma*norm.ppf(p))
        r = lambda p,mu: np.exp(mu + np.sqrt(2)*sigma*erfinv(2*p-1) + (erfinv(2*p-1))**2)*sigma*np.sqrt(2*np.pi)
        s = lambda p,mu: -0.5*np.exp(mu + (sigma**2)/2)*erf(sigma/np.sqrt(2)-erfinv(2*p-1))

        # Finding a root of this function corresponds to the optimal restart quantile.
        f = lambda p: (p-1)*q(p,0) + p*(1-p)*r(p,0) - s(p,0) + s(0,0)

        currentRes = 0
        vErr = False
        restart = 0
    
        try:
            # The root can be found be numerical methods such as bisection.
            restart = self.binSearch(f)
        except:
            print("Value Error occured!")
            vErr = True
        currentRes = restart
    
        # The optimal restart time is obtained by applying the quantile function to the optimal restart quantile.
        return q(currentRes,muIn)

    def binSearch(self, f, left = 0, right = 1, iterations = 15):
        """
            This function finds the root of a function by bisection.
        """
        
        val = right -(right-left)/2
        s = f(val)
        sleft = f(10**(-10) if left == 0 else left)
        if iterations == 0 or s == 0:
            return val

        if s * sleft > 0:
            return self.binSearch(f,left = left + (right-left)/2, right = right, iterations = iterations-1)
            
        return self.binSearch(f,left = left, right = right - (right-left)/2, iterations = iterations-1)
