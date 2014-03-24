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
def sigmoid(x, ex=0, inh=0, n=0, k=10, x0 = 0.5): # sigmoid activation function
    nz = 2*(0.5-np.rand())*n
    return 1/(1+np.exp(-k*(((ex+x)/2+nz-inh)-x0)))

res = 100
pltData = []
pltDataXX = []
pltDataX = []
e = 0.5
i = 0.
n = 0.
for st in range(res):
    x = (st+1.)/res
    for t in range(5):
        pltData.append(x)
    for t in range(15):
        print t, x
        pltDataX.append(x)
        pltDataXX.append(sigmoid(x,e,i,n))
        x = sigmoid(x,e,i,n)
        pltData.append(x)

plt.plot(pltData)
#plt.figure()
#plt.plot(pltDataX,pltDataXX)
plt.show()