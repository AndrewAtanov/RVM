'''Friedman #2 data example'''

import rvm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman2
from sklearn.datasets import make_friedman3

print('Friedman data set')

np.random.seed(2)
n_exp = 1
rel_vec = 0
RMSE = 0
for i in range(n_exp):
    train_x, train_y = make_friedman2(240, noise=0)
    cl = rvm.RVMRegression(kernel="rbf",  gamma=0.00001)
    cl.fit(np.matrix(train_x), np.matrix(train_y.reshape(240, 1)))
    rel_vec += len(cl.rel_ind)

    valid_x, valid_y = make_friedman2(240)
    pred_y = cl.predict(valid_x)

    RMSE += mean_squared_error(valid_y, pred_y) ** 0.5

print("Vectors: ", rel_vec / n_exp)
print("Root mean square = ", RMSE / n_exp)
