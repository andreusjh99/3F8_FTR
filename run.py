
#############imports

from __future__ import division
import numpy as np
#import warnings
#warnings.simplefilter("error", RuntimeWarning) #to ensure warnings are now exceptions

#np.seterr(under='print')

#import matplotlib.pyplot as plt
#import matplotlib.style
#matplotlib.style.use('classic')

#import scipy.optimize

#############functions
import misc

#########################################################################################################
#############Implementation

# We load the data
X = np.loadtxt('X.txt')
y = np.loadtxt('y.txt')

# We randomly permute the data
permutation = np.random.permutation(X.shape[ 0 ])
X = X[ permutation, : ]
y = y[ permutation ]

# We plot the data
misc.plot_data(X, y)

# We split the data into train and test sets
n_train = 800
X_train = X[ 0 : n_train, : ]
X_test = X[ n_train :, : ]
y_train = y[ 0 : n_train ]
y_test = y[ n_train : ]

####################################################### VANILLA ##################################################################

#misc.run_model(X, y, X_train, X_test, y_train, y_test, approach = "ML", rbf = False, alpha = 0.001, n_steps = 20)


####################################################### ML ##################################################################

#misc.run_model(X, y, X_train, X_test, y_train, y_test, approach = "ML", rbf = True, alpha = 0.004, n_steps = 700, l = 0.1)


####################################################### MAP ##################################################################

#misc.run_model(X, y, X_train, X_test, y_train, y_test, approach = "MAP", rbf = True, l = 0.1, var_w = 1.0)


####################################################### LAPLACE ##################################################################

#misc.run_model(X, y, X_train, X_test, y_train, y_test, approach = "Laplace", rbf = True, l = 0.1, var_w = 1.0)


####################################################### GRID SEARCH FOR LAPLACE ##################################################################
"""
var_w_arr = np.linspace(0.1, 4.0, 2)
l_arr = np.linspace(0.1, 0.5, 2)

model_ev_arr = misc.grid_search(var_w_arr, l_arr, X_train, y_train)

misc.plot_heatmap(var_w_arr, l_arr, model_ev_arr)
#"""

####################################################### OPTIMUM VAR_W AND L ##################################################################
#"""
prior_var_op, l_op = 0.3786, 0.3857

######### MAP

misc.run_model(X, y, X_train, X_test, y_train, y_test, approach = "MAP", rbf = True, l = l_op, var_w = prior_var_op)      


######### LAPLACE

misc.run_model(X, y, X_train, X_test, y_train, y_test, approach = "Laplace", rbf = True, l = l_op, var_w = prior_var_op) 
#"""     