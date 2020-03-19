#############imports

from __future__ import division
import numpy as np
#import warnings
#warnings.simplefilter("error", RuntimeWarning) #to ensure warnings are now exceptions

np.seterr(under='print')

import matplotlib.pyplot as plt
import matplotlib.style
matplotlib.style.use('classic')

import scipy.optimize

# Plots the points in 2D together with their labels
def plot_data_internal(X, y):
    """
    Inputs:
        X: 2d array with the input features
        y: 1d array with the class labels (0 or 1)
    Output: 
        2D matrices with the x and y coordinates of the points shown in the plot
    """
    
    x_min, x_max = X[ :, 0 ].min() - .5, X[ :, 0 ].max() + .5
    y_min, y_max = X[ :, 1 ].min() - .5, X[ :, 1 ].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    plt.figure()
    plt.xlim(xx.min(None), xx.max(None))
    plt.ylim(yy.min(None), yy.max(None))
    ax = plt.gca()
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label = 'Class 1')
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label = 'Class 2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Plot data')
    plt.legend(loc = 'upper left', scatterpoints = 1, numpoints = 1)
    return xx, yy

# Plots the data without returning anything by calling "plot_data_internal".
def plot_data(X, y):
    """
    Input:
        X: 2d array with the input features
        y: 1d array with the class labels (0 or 1)
    Output: Nothing.
    """
    
    xx, yy = plot_data_internal(X, y)
    plt.show()

########################################
# The logistic function
def logistic(x): return 1.0 / (1.0 + np.exp(-x))

#Computes the average loglikelihood of the logistic classifier on some data.
def compute_average_ll(X_tilde, y, w, training = True):
    """
    Input:
        X_tilde: matrix of input features for which to make predictions (N x M)
        y: vector of binary output labels (N x 1)
        w: vector of model parameters (M x 1)
        training: (bool) True when training, False when not, ie when calc model ev
    Output: 
        The average loglikelihood
    """
    
    N = X_tilde.shape[0]
    sigmoid = logistic(np.dot(X_tilde, w))
    
    try:
        ll = (np.dot(y, np.log(sigmoid)) + np.dot((1 - y), np.log(1.0 - sigmoid)))/N
    except: #due to runtime warning, 0 encountered in log.
        if training == False:    
            print('exception while calculating model evidence!')
        
        X_dot_w = np.dot(X_tilde, w)
        
        ll = 0
        for i in range(len(sigmoid)):
            if sigmoid[i] == 0.:
                ll += y[i]*X_dot_w[i]
            elif sigmoid[i] == 1.:
                ll += (1 - y[i])*(-X_dot_w[i])
            else:
                ll += y[i]*np.log(sigmoid[i]) + (1 - y[i])*np.log(1.0 - sigmoid[i])
        ll = ll/N
        
    return ll

# Expands a matrix of input features by adding a column equal to 1.
def get_x_tilde(X): 
    """
    Input:
        X: matrix of input features. (N x .)
    Output: 
        Matrix x_tilde with one additional constant column equal to 1 added. (N x .+1)
    """
    
    return np.concatenate((np.ones((X.shape[ 0 ], 1 )), X), 1)

# Finds the model parameters by optimising the likelihood using gradient descent
def fit_w_ml(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha):
    """
    Input:
        X_tilde_train: matrix of training input features (N x M)
        y_train: vector of training binary output labels (N x 1)
        X_tilde_test: matrix of test input features
        y_test: vector of test binary output labels 
        alpha: step_size_parameter for the gradient based optimisation
        n_steps: the number of steps of gradient based optimisation
    Output: 
        w: Vector of model parameters w (M x 1)
        ll_train: Vector with average log-likelihood values obtained on the training set (n_steps x 1)
        ll_test: Vector with average log-likelihood values obtained on the test set (n_steps x 1)
    """
    
    w = np.random.randn(X_tilde_train.shape[ 1 ]) #(3 x 1)
    ll_train = np.zeros(n_steps)
    ll_test = np.zeros(n_steps)
    for i in range(n_steps):
        sigmoid_value = logistic(np.dot(X_tilde_train, w))
		
		# Gradient-based update rule for w. To be completed by the student
        w = w + alpha*(np.dot(np.transpose(X_tilde_train), (y_train - sigmoid_value)))
        
        ll_train[ i ] = compute_average_ll(X_tilde_train, y_train, w)
        ll_test[ i ] = compute_average_ll(X_tilde_test, y_test, w)
		
    return w, ll_train, ll_test

###########################################
# Gets confusion matrix based on test data
def classify(X_tilde_test, y_test, w, laplace = False, var_w = 0):
    """
    Input:
        X_tilde_test: matrix of test input features (N x M)
        y_test: vector of training binary output labels (N x 1)
        w: Vector of model parameters w (M x 1)
        laplace: Bool (whether it's ML/MAP(False) or Laplace(True))
        var_w: prior variance. 0 for ML/MAP, float for Laplace
    Output: 
        true_neg/total0: proportion of true negatives (float)
        false_pos/total0: proportion of false positives (float)
        false_neg/total1: proportion of false negatives (float)
        true_pos/total1: proportion of true positives (float)
    """
    
    if laplace:
        output_prob = calc_lap_pred_distn(X_tilde_test, w, var_w)
    else:
        sigmoid = logistic(np.dot(X_tilde_test, w))
        output_prob = sigmoid
	
    true_neg, false_neg, false_pos, true_pos = 0, 0, 0, 0
	
    for i in range(len(output_prob)):
		#classify
        if output_prob[i] > 0.5:
            y_hat = 1
        else:
            y_hat = 0
		
		#idenify pos and neg
        if y_test[i] == 0:
            if y_hat == 0:
                true_neg += 1
            else: 
                false_pos += 1
        else:
            if y_hat == 0:
                false_neg += 1
            else: 
                true_pos += 1
	
    total0 = true_neg + false_pos #total 0s detected
    total1 = false_neg + true_pos #total 1s detected
    total = total0 + total1
    assert total == len(y_test)
	
    return true_neg/total0, false_pos/total0, false_neg/total1, true_pos/total1

# Plots the average log-likelihood for the entire training after the training.
def plot_ll(ll):
    """
    Input:
        ll: vector of log-likelihoods
    Output: Nothing
    """
    
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, len(ll) + 2)
    plt.ylim(min(ll) - 0.1, max(ll) + 0.1)
    ax.plot(np.arange(1, len(ll) + 1), ll, 'r-')
    plt.xlabel('Steps')
    plt.ylabel('Average log-likelihood')
    plt.title('Plot Average Log-likelihood Curve')
    plt.show()
    
# Plots the predictive probabilities of the logistic classifier
def plot_predictive_distribution(X, y, w, map_inputs = lambda x : x, var_w = 0, laplace = False):
    """
    Input:
        X: ORIGINAL 2d array with the input features for the data
        y: 1d array with the class labels (0 or 1) for the data
        w: parameter vector
        map_inputs: function that expands the original 2D inputs using basis functions.
        var_w: prior variance (set at 0 when laplace == False)
        laplace: bool (sigmoid prob (False) and Bayesian, ie Laplace (True))
    Output: Nothing.
    """
    
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_tilde = get_x_tilde(map_inputs(np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1)))
    
    if laplace == False:
        Z = logistic(np.dot(X_tilde, w))
    else:
        Z = calc_lap_pred_distn(X_tilde, w, var_w)
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 4)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
    plt.show()

######################################
#Replaces initial input features by evaluating Gaussian basis functions on a grid of points
def evaluate_basis_functions(l, X, Z):
    """
    Inputs:
        l: hyper-parameter for the width of the Gaussian basis functions (float)
        Z: location of the Gaussian basis functions
        X: points at which to evaluate the basis functions
    Output: 
        Feature matrix with the evaluations of the Gaussian basis functions.
    """
    
    X2 = np.sum(X**2, 1)
    Z2 = np.sum(Z**2, 1)
    ones_Z = np.ones(Z.shape[ 0 ])
    ones_X = np.ones(X.shape[ 0 ])
    r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
    return np.exp(-0.5 / l**2 * r2)

######################################
# Evaluates the objective/cost fn for MAP
def cost(w, X_tilde, y, var_w):
	
    """
    Input:
        w: parameter vector (M x 1)
        X_tilde: matrix of input features (N x M)
        y: 1d array with the class labels (0 or 1) for the data (N x 1)
        var_w: prior variance (float)
    Output: -lp: negative of log posterior
    """
    
    N = X_tilde.shape[0]
    ll = compute_average_ll(X_tilde, y, w)*N
        
    lp = ll - 0.5*np.dot(w, w)/var_w #log posterior
        
    return -lp

# Finds the model parameters by optimising the likelihood using gradient descent
def fit_w_map(X_tilde_train, y_train, var_w):
    """
    Input:
        X_tilde_train: matrix of training input features (N x M)
        y_train: vector of training binary output labels (N x 1)
        var_w = prior variance (float)
    Output: 
        w_map: Vector of model parameters w (M x 1)
    """
    
    w = np.random.randn(X_tilde_train.shape[1])
    assert w.shape[0] == X_tilde_train.shape[1]
	
    w_map = scipy.optimize.fmin_l_bfgs_b(cost, w, args = (X_tilde_train, y_train, var_w), approx_grad=True)[0]

    return w_map

####################################
# Computes the negative of Hessian, A
def compute_A(X_tilde, w_map, var_w):
    """
    Inputs:
        X_tilde: matrix of input features (N x M)
        w_map: vector of MAP weights (M x 1)
        var_w: prior var (float)
    Output:
        A
    """
    
    sigmoid = logistic(np.dot(X_tilde, w_map))
    
    A = np.identity(X_tilde.shape[1])/var_w
    for i in range(X_tilde.shape[0]):
        A += sigmoid[i]*(1 - sigmoid[i])*np.outer(X_tilde[i], X_tilde[i])
	
    assert A.shape[0] == A.shape[1] #(M x M)
    assert A.shape[0] == X_tilde.shape[1]
	
    return A
	
# Evaluates log of model evidence
def model_evidence(X_tilde, y, w_map, var_w):
    """
    Inputs:
        X_tilde: matrix of input features (N x M)
        y: binary class (N x 1)
        w_map: vector of MAP weights (M x 1)
        var_w: prior var (float)
    Output:
        model_ev: log of model evidence (float)
    """
    
    M = X_tilde.shape[1]
    N = X_tilde.shape[0]
    A = compute_A(X_tilde, w_map, var_w)
    
    ll_ev = N*compute_average_ll(X_tilde, y, w_map, training = False)
	
	#cholesky decomposition to prevent numerical errors
    chol = np.linalg.cholesky(A)
    log_det_A = 2*np.sum(np.log(np.diag(chol)))
	
    prior_ev = -0.5*M*np.log(var_w) - 0.5*np.dot(w_map, w_map)/var_w - 0.5*log_det_A
	
    model_ev = ll_ev + prior_ev
	
    return model_ev

# Performs Laplace approximation on predictive probability distn
def calc_lap_pred_distn(X_tilde, w_map, var_w):
    """
    Inputs:
        X_tilde: matrix of input features (N x M)
        w_map: vector of MAP weights (M x 1)
        var_w: prior var (float)
    Output:
        pred_distn: vector of predictive prob (N x 1)
    """
    
    mu_a = np.dot(X_tilde, w_map) #(N x 1)
	
    A = compute_A(X_tilde, w_map, var_w)
	
    A_inv = np.linalg.inv(A)
    var_a = np.dot(np.dot(X_tilde, A_inv), np.transpose(X_tilde)) #(N x N)
    var_a = np.diag(var_a)

    assert mu_a.shape[0] == var_a.shape[0]
    assert mu_a.shape[0] == X_tilde.shape[0]
	
	#Approximate logistic in integral with probit
	#This gives another probit and is approximated with a logistic
    k_var_a = (1 + np.pi*var_a/8)**-0.5
	
    pred_distn = logistic(k_var_a * mu_a)
	
    return pred_distn

###########################################
# Performs grid search
def grid_search(var_w_arr, l_arr, X_train, y_train):
    
    """
    Inputs:
        var_w_arr: array of prior var (P x 1)
        l_arr: array of l (Q x 1)
        X_train: original input feature matrix
        y_train: binary label (N x 1)
    Output:
        model_ev_arr: matrix of model evidence (P x Q)
    """

    #initialise grid with zeros
    s = (len(var_w_arr), len(l_arr))
    model_ev_arr = np.zeros(s)
    
    k = 1
    
    print(">>>>>> Start grid search >>>>>>")
    #update grid
    for i in range(len(var_w_arr)):
        for j in range(len(l_arr)):
            print("{}. Running model for (var_w, l) = ({}, {}) ...".format(k, var_w_arr[i], l_arr[j]))
            X_tilde_train = get_x_tilde(evaluate_basis_functions(l_arr[j], X_train, X_train))
            
            w_map = fit_w_map(X_tilde_train, y_train, var_w_arr[i])
            model_ev = model_evidence(X_tilde_train, y_train, w_map, var_w_arr[i])
    
            model_ev_average = model_ev/(X_tilde_train.shape[0])
            model_ev_arr[i][j] = model_ev_average
            
            k += 1
    
    print("Done!")
    
    assert k == len(var_w_arr)*len(l_arr) + 1
    
    return model_ev_arr
	
# Plots heatmap for grid_search
def plot_heatmap(var_w_arr, l_arr, model_ev_arr, cbarlabel=''):
    """
    Inputs:
        var_w_arr: array of prior variance (P x 1)
        l_arr: array of l (Q x 1)
        model_ev_arr: array of model evidence (P x Q)
    Output: Nothing
	"""
    fig, ax = plt.subplots()
    im = ax.imshow(model_ev_arr, cmap='inferno', interpolation='nearest')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    
    # We want to show all ticks...
    l_arr1 = [round(l_arr[i], 4) for i in range(len(l_arr))]
    var_w_arr1 = [round(var_w_arr[i], 4) for i in range(len(var_w_arr))]
    
    ax.set_xlabel('l (width of rbf gaussian)')
    ax.set_ylabel('prior var')
    ax.set_xticks(np.arange(len(l_arr1)))
    ax.set_yticks(np.arange(len(var_w_arr1)))
	# ... and label them with the respective list entries
    ax.set_xticklabels(l_arr1)
    ax.set_yticklabels(var_w_arr1)

	# Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

    ax.set_title("Average training model evidence")
    fig.tight_layout()
    
    plt.show()


# Runs model
def run_model(X, y, X_train, X_test, y_train, y_test, approach = "ML", rbf = False, alpha = 0.001, n_steps = 20, l = 0.1, var_w = 1.0):
    
    """
    Inputs:
        X: original input feature matrix (full)
        y: binary label (full)
        X_train: original input training feature matrix
        X_test: original input test feature matrix
        y_train: binary label for training data
        y_test: binary label for test data
        approach: (string) "ML", "MAP" or "Laplace"
        rbf: (bool) False for no feature expansion, True for feature expansion
        
        alpha: (float) learning rate for ML
        n_steps: (int) number of training steps for ML
        
        l: (float) width of rbf
        
        var_w: (float) prior variance for MAP and Laplace
    Output: Nothing
    """
    
    # expand input or not
    if rbf == False:
        X_tilde_train = get_x_tilde(X_train)
        X_tilde_test = get_x_tilde(X_test)
    else:
        # We expand the data
        l = l

        X_tilde_train = get_x_tilde(evaluate_basis_functions(l, X_train, X_train))
        X_tilde_test = get_x_tilde(evaluate_basis_functions(l, X_test, X_train))

    # splitting into the 3 approaches
    if approach == "ML":
        print(">>>>>>>>>>>>>>> ML >>>>>>>>>>>>>>>")
        
        # We train the classifier
        alpha = alpha #0.004 for rbf
        n_steps = n_steps #700 for rbf
        
        print("Fitting weights ...")
        w, ll_train, ll_test = fit_w_ml(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha)
    
        # We plot the training and test log likelihoods
        print(">>>>>>>>> TRAINING CURVE")
        plot_ll(ll_train)
        print("")
        print(">>>>>>>>> TEST CURVE")
        plot_ll(ll_test)
    
        # We print the final av log-likelihoods
        ll_train_f = ll_train[-1]
        ll_test_f = ll_test[-1]
        print("ll_train_f = {}, ll_test_f = {}".format(ll_train_f, ll_test_f))

        # We output the pos and neg
        true_neg, false_pos, false_neg, true_pos = classify(X_tilde_test, y_test, w)
        print("true_neg = {}, false_pos = {}, false_neg = {}, true_pos = {}".format(true_neg, false_pos, false_neg, true_pos))
    
        # We plot the predictive distribution
        if rbf == False:
            plot_predictive_distribution(X, y, w)
        else:
            plot_predictive_distribution(X, y, w, lambda x : evaluate_basis_functions(l, x, X_train))
            
    elif approach == "MAP":
        print(">>>>>>>>>>>>>>> MAP >>>>>>>>>>>>>>>")
        
        var_w = var_w

        # We train the classifier
        print("Fitting weights ...")
        w_map = fit_w_map(X_tilde_train, y_train, var_w)

        # We print the final av log-likelihoods
        ll_train_f = compute_average_ll(X_tilde_train, y_train, w_map)
        ll_test_f = compute_average_ll(X_tilde_test, y_test, w_map)
        print("ll_train_f = {}, ll_test_f = {}".format(ll_train_f, ll_test_f))

        # We output the pos and neg
        true_neg, false_pos, false_neg, true_pos = classify(X_tilde_test, y_test, w_map)
        print("true_neg = {}, false_pos = {}, false_neg = {}, true_pos = {}".format(true_neg, false_pos, false_neg, true_pos))

        # We plot the predictive distribution
        if rbf == False:
            plot_predictive_distribution(X, y, w_map)
        else:
            plot_predictive_distribution(X, y, w_map, lambda x : evaluate_basis_functions(l, x, X_train))
            
    elif approach == "Laplace":
        print(">>>>>>>>>>>>>>> LAPLACE >>>>>>>>>>>>>>>")
        
        var_w = var_w
        
        # We train the classifier
        print("Fitting weights ...")
        w_map = fit_w_map(X_tilde_train, y_train, var_w)

        # We print the final av log-likelihoods
        ll_train_f = model_evidence(X_tilde_train, y_train, w_map, var_w)/X_tilde_train.shape[0]
        ll_test_f = model_evidence(X_tilde_test, y_test, w_map, var_w)/X_tilde_test.shape[0]
        print("ll_train_f = {}, ll_test_f = {}".format(ll_train_f, ll_test_f))

        # We output the pos and neg
        true_neg, false_pos, false_neg, true_pos = classify(X_tilde_test, y_test, w_map, laplace = True, var_w = var_w)
        print("true_neg = {}, false_pos = {}, false_neg = {}, true_pos = {}".format(true_neg, false_pos, false_neg, true_pos))
        
        print("Plotting ... (This might take a while.)")
        # We plot the predictive distribution
        if rbf == False:
            plot_predictive_distribution(X, y, w_map, var_w, laplace = True)
        else:
            plot_predictive_distribution(X, y, w_map, lambda x : evaluate_basis_functions(l, x, X_train), var_w, laplace = True)
        
    else:
        print("Approach can only be ML, MAP or Laplace")
        
    print("DONE!")