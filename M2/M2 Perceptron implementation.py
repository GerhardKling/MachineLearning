"""
Implementing the perceptron learning rule
@author: GK
"""

import numpy as np
from matplotlib import pyplot as plt 

#=========================================================================
#Example
#=========================================================================
#Three cases and two dimensions (constant term)
#j = 1
case_1 = np.array([1, 1, 2])

#j = 2
case_2 = np.array([1, 2, 4])

#j = 3
case_3 = np.array([1, 3, 1])

#Array of actual output
y = np.array([1, 1, -1])

#Create data matrix X
#Transpose input column vectors
reshape_1 = case_1[None, :]

reshape_2 = case_2[None, :]

reshape_3 = case_3[None, :]

#Add to data matrix X

X = np.vstack((reshape_1, reshape_2, reshape_3))
    
#Initial weights
w = np.zeros(3)

#Learning rate
eta = 0.5

#Initial prediction
z = np.matmul(X, w)
y_hat = np.where(z >= 0, 1, -1)

#Learning rule
#First case: no adjustment
w += eta * (y[0] - y_hat[0]) * X[0, :]

#Second case: no adjustment
w += eta * (y[1] - y_hat[1]) * X[1, :]

#Third case: adjustment
w += eta * (y[2] - y_hat[2]) * X[2, :]

#Then go back to line 43 and repeat
#Stop once no adjustments are needed
#Set maximum number of iterations


#=========================================================================
#Loop
#=========================================================================
#Convert the logic in lines 42 to 58 into a loop

#Set maximum iterations
max_iter = 100
i = 0

#Initial weights
w = np.zeros(3)

#Changing initial condition matters
# w = np.array([0.1, 0.3, 0.4])

while i < max_iter:
    z = np.matmul(X, w) #net input
    y_hat = np.where(z >= 0, 1, -1) #prediction
    w += eta * (y[0] - y_hat[0]) * X[0, :]
    print(w)
    w += eta * (y[1] - y_hat[1]) * X[1, :]
    print(w)
    w += eta * (y[2] - y_hat[2]) * X[2, :]
    print(w)
    i += 1
    
print(w)

x0_constant = np.ones(100)
x1_axis = np.linspace(0, 10, 100)

#z = w_0 + w_1x_1 + w_2x_2 = 0
#Solve for x_2
decision_boundary = 1/w[2] * (- w[0] - w[1] * x1_axis)

plt.figure(1)
plt.plot(x1_axis, decision_boundary)
plt.plot(X[:,1], X[:, 2], 'o', color='black')


#=========================================================================
#Refactoring
#=========================================================================
#Stop if you do not make any errors
#Set maximum iterations
max_iter = 100
i = 0

#Initial weights
w = np.zeros(3)

while i < max_iter:
    z = np.matmul(X, w) #net input
    y_hat = np.where(z >= 0, 1, -1) #prediction
    w += eta * (y[0] - y_hat[0]) * X[0, :]
    print(w)
    w += eta * (y[1] - y_hat[1]) * X[1, :]
    print(w)
    w += eta * (y[2] - y_hat[2]) * X[2, :]
    print(w)
    i += 1
    error = sum(abs(y - y_hat))
    if error == 0:
        break
    
print(w)


#Define a function
def perceptron_rule(X, y, max_iter: int, eta: float):
    """
    Parameters
    ----------
    X : Data matrix with N cases (rows) and m inputs (columns)
    y : Column vector of actual output for N cases
    max_iter : int: maximum number of iterations
    eta : float: learning rate
    
    Returns
    -------
    w : weights
    """
    #Initial weights
    np.random.seed(1)
    w = np.random.rand(3)
    i = 0
    while i < max_iter:
        z = np.matmul(X, w) #net input
        y_hat = np.where(z >= 0, 1, -1) #prediction
        #Adjustment of w
        for n in range(X.shape[0]):
            w += eta * (y[n] - y_hat[n]) * X[n, :]
        #Check for errors
        z = np.matmul(X, w) #net input
        y_hat = np.where(z >= 0, 1, -1)
        error = sum(abs(y - y_hat))
        if error == 0:
            print(f'Solution found: {error}')
            break
        else:
            i += 1
    return w

   
w = perceptron_rule(X, y, 10, 0.5)  
z = np.matmul(X, w)
y_hat = np.where(z >= 0, 1, -1)   
error = sum(abs(y - y_hat))
decision_boundary_new = 1/w[2] * (- w[0] - w[1] * x1_axis) 

plt.figure(2)
plt.plot(x1_axis, decision_boundary_new)
plt.plot(X[:,1], X[:, 2], 'o', color='black')    
    
 
 
    
    
    
    
    
    
