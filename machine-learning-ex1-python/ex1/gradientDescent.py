import numpy as np;
import computeCost;
def gradientDescent(X, y, theta, alpha, num_iters):
#GRADIENTDESCENT Performs gradient descent to learn theta
#   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
#   taking num_iters gradient steps with learning rate alpha

# Initialize some useful values
    m = y.size; # number of training examples
    J_history = np.zeros(num_iters);

    for iter in range(0,num_iters):

    # ====================== YOUR CODE HERE ======================
    # Instructions: Perform a single gradient step on the parameter vector
    #               theta. 
    #
    # Hint: While debugging, it can be useful to print out the values
    #       of the cost function (computeCost) and gradient here.
    #
    # Vectorized
    
        theta = theta - alpha/m * np.dot(X.transpose(),(np.dot(X,theta)-y))

        # ============================================================

        # Save the cost J in every iteration    
        J_history[iter] = computeCost.computeCost(X, y, theta);
    return theta;