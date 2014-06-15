# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:52:36 2013

@author: Brutsev

visualize sigmoid mapping
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as np
from matplotlib import cm


def sigm(x): # sigmoid activation function
    return 1/(1+np.exp(10*(x-0.5)))
def sigmoid(x, ex=0, inh=0, n=0, k=10, x0 = 0.5): # sigmoid activation function
    nz = 2*(0.5-np.rand())*n
    return 1/(1+np.exp(-k*(((ex+x)+nz-inh)-x0)))

res = 1
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
        #pltDataX.append(x)
        #pltDataXX.append(sigmoid(x,e,i,n))
        x = sigmoid(x,e,i,n)
        pltData.append(x)

#plt.plot(pltData)
#plt.figure()
#plt.plot(pltDataX,pltDataXX)

res =300
bifSetk = []
bifSetx0 = []
bifSetx = []
for ki in range(1):
    kk = 10
    for x0i in range(res):
        xx0 = (x0i+0.)/res
        #print 'k:',kk,' x0:',xx0
        for xi in range(res):
            x = (xi+0.)/res
#            if (kk==10):
#                print 'x=',x,'y=',sigmoid(x, k=kk, x0=xx0)
            if (np.absolute(x-sigmoid(x, k=kk, x0=xx0))<(1./res)):
                bifSetk.append(kk)
                bifSetx0.append(xx0)
                bifSetx.append(x)
                print '!!! x=',x
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter(bifSetk,bifSetx0,bifSetx)
plt.scatter(bifSetx0,bifSetx)

plt.show()