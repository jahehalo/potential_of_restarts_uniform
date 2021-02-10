import os
import _pickle
from scipy.stats import norm
from scipy.special import erf
from scipy.special import erfinv
from scipy.special import gamma
from scipy.special import gammainc
from scipy.optimize import fmin
import numpy as np
from keras import backend as K
import math
import random
import warnings
import sys
from scipy.stats import exponweib
from scipy.stats import genpareto
from scipy.stats import lognorm
from subprocess import Popen, PIPE



def calculate_mean(restart, observed):
    """
        This function calculates the (empirical) speedup achived by a restart time.
    """
    if np.isinf(restart):
        return 1.0 # no speedup and no slowdown if there are no restarts. 
    restart = int(restart)
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
    # Werte größer als 1 sind ein tatsächlicher Speedup.

    return np.mean(observed)/result


def binSearch(f, left = 0, right = 1, iterations = 15):
    """
        This method is used to find the optimal restart quantile.
    """
    val = right -(right-left)/2
    s = f(val)
    sleft = f(10**(-10) if left == 0 else left)
    if iterations == 0 or s == 0:
        return val

    if s * sleft > 0:
        return binSearch(f,left = left + (right-left)/2, right = right, iterations = iterations-1)
        
    return binSearch(f,left = left, right = right - (right-left)/2, iterations = iterations-1)

def weibullRestartTime(a,b,loc):
    """
        This function is used to find the optimal restart time for the Weibull distribution.
    """

    mu = loc
    k = b

    # function q is the quantile function, r is the derivative of the quantile function and s is the anti-derivative.
    q = lambda p,m: m + a*np.power((-np.log(1-p)),1/k)
    r = lambda p: a*np.power((-np.log(1-p)),(1/k)-1)/(k*(1-p))
    s = lambda p: a * gamma(1+(1/k))*gammainc(1+(1/k),-np.log(1-p))   

    if b >= 1.0:
        # Restarts are not useful if the shape is greater than 1.
        return np.inf
 
    if loc/a < 1e-6:
        # To avoid numerical issues...
        return loc
    
    # The optimal restart quantile is the root of the following function...
    f = lambda p: 0 if p == 0 else gamma(1+(1/k))*gammainc(1+(1/k),-np.log(1-p)) + (1-p)*np.power(-np.log(1-p),1/k) - (p/k)*np.power(-np.log(1-p),(1/k)-1) + (mu/a)
    

    currentRes = 0
    vErr = False
    restart = 0
    
    # The optimal restart quantile can be found by numerical means, for example bisection.
    restart = binSearch(lambda x: -f(x))
    currentRes = restart
    if vErr:
        print("Value Error occured, result is questionable")

    
    # To obtain the optimal restart time from the optimal restart quantile, apply the quantile function.
    return q(currentRes,mu)


def lognormRestartTime(muIn,sigma,maxIterations = 100, errorBoundary = 1.0e-10):
    """
        This function is used to find the optimal restart time for the lognormal distribution.
    """
    if sigma < 0.7: 
        """
            In theory restarts are always useful for lognormal distributions.
            However, for sigma < 0.7, the optimal restart times are so high that they do not influence the expected runtime in practice.
        """
        return np.inf   

    # function q is the quantile function, r is the derivative of the quantile function and s is the anti-derivative.
    q = lambda p,mu: np.exp(mu + sigma*norm.ppf(p))
    r = lambda p,mu: np.exp(mu + np.sqrt(2)*sigma*erfinv(2*p-1) + (erfinv(2*p-1))**2)*sigma*np.sqrt(2*np.pi)
    s = lambda p,mu: -0.5*np.exp(mu + (sigma**2)/2)*erf(sigma/np.sqrt(2)-erfinv(2*p-1))

    # The optimal restart quantile is the root of the following function...
    f = lambda p: (p-1)*q(p,0) + p*(1-p)*r(p,0) - s(p,0) + s(0,0)


    currentRes = 0
    vErr = False
    restart = 0


    try:
        # The optimal restart quantile can be found by numerical means, for example bisection.
        restart = binSearch(f)
    except:
        print("Value Error occured!")
        vErr = True
    currentRes = restart

    # To obtain the optimal restart time from the optimal restart quantile, apply the quantile function.
    return q(currentRes,muIn)



def scale_calc_weib(p):
    """
        The outputs of the neural networks are scaled to (0,1).
        This function rescales such an output to obtain the scale parameter of a Weibull distribution.
    """
    mu =  2.863891478217836   
    sigma = 0.146727246649470
    return np.exp(np.exp(mu + sigma*norm.ppf(p)))

def shape_calc_weib(p):
    """
        The outputs of the neural networks are scaled to (0,1).
        This function rescales such an output to obtain the shape parameter of a Weibull distribution.
    """
    alpha = 0.968511138239881
    c=17.264902910756167
    k=1.082668010912716
    return alpha*np.power(np.power(1/(1-p), 1/k)-1, 1/c)

def scale_calc_logn(p):
    """
        The outputs of the neural networks are scaled to (0,1).
        This function rescales such an output to obtain the scale parameter of a lognormal distribution.
    """
    mu=   2.686333243803358
    sigma=   0.090459464555333 
    return np.exp(mu + sigma*norm.ppf(p))

def shape_calc_logn(p):
    """
        The outputs of the neural networks are scaled to (0,1).
        This function rescales such an output to obtain the shape parameter of a lognormal distribution.
    """
    alpha= 0.912047580559537
    c=  10.966647985287899
    k=   0.705838374748621 
    return alpha*np.power(np.power(1/(1-p), 1/k)-1, 1/c)
    
def startSolving(cnfPath, restart, seed, debug, nVar):
    """
        This function calls probSAT with the appropriate restart strategy.
    """
    random.seed(a=seed)
    output = None

    if restart == 'luby':
        p = Popen(['./probSAT/probSAT','-L ' + str(20*nVar),cnfPath,str(seed)],stdin = PIPE, stdout = PIPE, stderr=PIPE)
        output, err = p.communicate()
        rc = p.returncode
    elif restart == np.inf:
        # In this case no restarts are used.
        # probSAT is invoked with a timeout of 100000000000 flips.
        p = Popen(['./probSAT/probSAT','-q 100000000000',cnfPath,str(seed)],stdin = PIPE, stdout = PIPE, stderr=PIPE)
        output, err = p.communicate()
        rc = p.returncode
    else:
        # Here, the fixed-cutoff strategy is used.
        # probSAT is invoked with a timeout of 100000000000 flips.
        p = Popen(['./probSAT/probSAT','-m '+str(restart),'-q 100000000000', cnfPath,str(seed)],stdin = PIPE, stdout = PIPE, stderr=PIPE)
        output, err = p.communicate()
        rc = p.returncode

    return output.decode("utf-8")[:-1] + " " + str(restart)
    

def loadClf(path):
    with open(path,'rb') as f:
        return _pickle.load(f)
