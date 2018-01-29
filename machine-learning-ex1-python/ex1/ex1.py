# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:46:16 2018

@author: KAIJIA
"""
## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     warmUpExercise.py
#     plotData.py
#     gradientDescent.py
#     computeCost.py
#     gradientDescentMulti.py
#     computeCostMulti.py
#     featureNormalize.py
#     normalEqn.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#
import matplotlib.pyplot as plt;
from matplotlib import cm;
import pandas as pd;
import numpy as np;
from numpy import genfromtxt;
import warmUpExercise;
import computeCost;
import gradientDescent;
#import plotData as plotData;

np.set_printoptions(precision=2)

## ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m
print('Running warmUpExercise ... \n');
print('5x5 Identity Matrix: \n');
A=warmUpExercise.warmUpExercise()
print (A)
input("Press the <ENTER> key to continue...")

##======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
my_data= genfromtxt("E:\Private\ML\machine-learning-ex1-python\ex1\ex1data1.txt",delimiter=',')
data=pd.read_csv('E:\Private\ML\machine-learning-ex1-python\ex1\ex1data1.txt')
X=my_data[:,0]
y=my_data[:,1]
m=y.shape

# Plot Data
# Note: You have to complete the code in plotData.m
#plotData.plotData(X,y)
plt.scatter(X,y,marker='*')
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()
input("Press the <ENTER> key to continue...")

## =================== Part 3: Cost and Gradient descent ===================

X=[np.ones(m),X]; # Add a column of ones to x
theta = np.zeros(2); # initialize fitting parameters

# convert from array to matrix
X=np.mat(X)
y=np.mat(y)
theta = np.mat(theta)

X=X.transpose()
y=y.transpose()
theta=theta.transpose()

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;

print('\nTesting the cost function ...\n')

# compute and display initial cost
J = computeCost.computeCost(X, y, theta);
print('With theta = [0 ; 0]\nCost computed = {0:.2f}\n'.format(J[0,0]))
print('Expected cost value (approx) 32.07\n');

# further testing of the cost function
J =  computeCost.computeCost(X, y, np.matrix([[-1],[2]]));
print('\nWith theta = [-1 ; 2]\nCost computed = {0:.2f}\n'.format(J[0,0]));
print('Expected cost value (approx) 54.24\n');

input("Program paused. Press the <ENTER> key to continue...")

print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta= gradientDescent.gradientDescent(X, y, theta, alpha, iterations);
#
# print theta to screen
print('Theta found by gradient descent:\n');
print(theta);
print('Expected theta values (approx)\n');
print(' -3.6303\n  1.1664\n\n');

# Plot the linear fit
#plt.hold(True); # keep previous plot visible
plt.scatter(my_data[:,0],my_data[:,1],marker='*',label='Training data')
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], X*theta, '-', label='Linear regression')
plt.legend()
plt.show()
#
# Predict values for population sizes of 35,000 and 70,000
np.set_printoptions(precision=6)
predict1 = np.matrix([[1,3.5]]) *theta;
print('For population = 35,000, we predict a profit of #f\n',predict1[0,0]*10000);
predict2 = [1, 7] * theta;
print('For population = 70,000, we predict a profit of #f\n',predict2[0,0]*10000);
#
input("Program paused. Press the <ENTER> key to continue...")

## ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100);
theta1_vals = np.linspace(-1, 4, 100);
s = (len(theta0_vals), len(theta1_vals))

# initialize J_vals to a matrix of 0's
J_vals = np.zeros(s);

# Fill out J_vals
for i in range(0,len(theta0_vals)):
    for j in range(0,len(theta1_vals)):
        t=np.matrix([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i,j] = computeCost.computeCost(X, y, t)

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals,cmap=cm.coolwarm,linewidth=0, antialiased=False)

#Show the plot
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show()

input("Program paused. Press the <ENTER> key to continue...")

## Contour plot
fig = plt.figure()
## Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(theta0_vals, theta1_vals, J_vals.transpose(), np.logspace(-2, 3, num=20))
plt.xlabel('theta_0'); 
plt.ylabel('theta_1');
plt.plot(theta[0], theta[1], 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plt.show()




