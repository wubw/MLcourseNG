# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:11:41 2018

@author: KAIJIA
"""
import matplotlib.pyplot as plt;
def plotData(x,y):
    plt.scatter(x,y,marker='*')
    plt.xlabel('Population of city in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()
    