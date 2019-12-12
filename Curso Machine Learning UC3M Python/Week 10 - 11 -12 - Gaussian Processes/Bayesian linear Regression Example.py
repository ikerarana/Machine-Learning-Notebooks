# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

wa= 1.53
wb = 0.2
b = -0.5

#Generate training set



x = np.arange(-2,2,0.5)

y = wa*x+wb*(x**2)+b + 1 * np.random.randn(x.shape[0])


plt.plot(x,y,'ob',ms=5,label = 'Training Data')



# BLR

#add intercept 

Xmat = np.vstack((np.ones(x.shape[0]),x))

sigma_z = 0.5

V_0 = np.eye(2)

prec_post = np.linalg.inv(V_0) + 1/sigma_z * Xmat@Xmat.T

cov_post = np.linalg.inv(prec_post)

mean_post = (1/sigma_z * cov_post @ Xmat @ y).reshape([1,-1])



## Test

x_new = np.arange(-10,10,0.05)

Xmat_new = np.vstack((np.ones(x_new.shape[0]),x_new))

mean_y_new = (mean_post @ Xmat_new).reshape(x_new.shape[0])
cov_y_new = np.diag(sigma_z + Xmat_new.T @ cov_post @ Xmat_new)




plt.plot(x,y,'ob',ms=5,label = 'Training Data')

plt.plot(x_new,mean_y_new,'-r',label = 'Posterior Mean')

plt.plot(x_new,mean_y_new+2*np.sqrt(cov_y_new),'--k',label = 'Posterior Mean + 2 std')

plt.plot(x_new,mean_y_new-2*np.sqrt(cov_y_new),'--g',label = 'Posterior Mean - 2 std')

plt.legend()

plt.xlabel('x')

plt.ylabel('y')