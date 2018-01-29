import numpy as np;
def featureNormalize(X):
#FEATURENORMALIZE Normalizes the features in X 
#   FEATURENORMALIZE(X) returns a normalized version of X where
#   the mean value of each feature is 0 and the standard deviation
#   is 1. This is often a good preprocessing step to do when
#   working with learning algorithms.

# You need to set these values correctly
    X_norm = X;
    mu = np.zeros(X.ndim);
    sigma = np.zeros(X.ndim);

# ====================== YOUR CODE HERE ======================
# Instructions: First, for each feature dimension, compute the mean
#               of the feature and subtract it from the dataset,
#               storing the mean value in mu. Next, compute the 
#               standard deviation of each feature and divide
#               each feature by it's standard deviation, storing
#               the standard deviation in sigma. 
#
#               Note that X is a matrix where each column is a 
#               feature and each row is an example. You need 
#               to perform the normalization separately for 
#               each feature. 
#
# Hint: You might find the 'mean' and 'std' functions useful.
#       
# return dimensions of this matrix
    n = X.ndim;

    for j in range(1,n):
        mu[j] = np.mean(X[:,j]);
        sigma[j] = np.std(X[:,j],ddof=1);
  
        X_norm [:,j] = (X_norm[:,j] - np.mean(X[:,j]))/np.std(X[:,j],ddof=1); # ddof=1 to get the same value as matlab

