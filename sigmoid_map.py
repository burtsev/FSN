# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:52:36 2013

@author: Brutsev

visualize sigmoid mapping
"""
import matplotlib.pyplot as plt
import scipy as np


def sigm(x): # sigmoid activation function    
    return 1/(1+np.exp(10*(x-0.5)))
def sigmoid(x, ex=0, inh=0, n=0, k=10): # sigmoid activation function
    nz = 2*(0.5-np.rand())*n
    return 1/(1+np.exp(-k*((ex+nz+x)/2-inh-0.5)))

res = 10
pltData = []
e = 0.4
i = 0.
n = 0.05
for st in range(res):
    x = (st+1.)/res
    for t in range(5):
        pltData.append(x)
    for t in range(10):
        print t, x
        x = sigmoid(x,e,i,n)
        pltData.append(x)

plt.plot(pltData)
plt.show()