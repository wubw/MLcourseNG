import numpy as np;
def computeCostMulti(X, y, theta):
#COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
#   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y

# Initialize some useful values
    m =  y.size; # number of training examples

# You need to return the following variables correctly 
    J = 0;

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.

# Vectorized

    J= 1/(2*m)*sum(np.square(np.dot(X,theta)-y));

    return J

