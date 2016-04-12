import rvm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

n_exp = 100
rel_vec = 0
RMSE = 0
for i in range(n_exp):
    train_x = np.matrix(np.random.uniform(-10, 10, size=(100, 2)))
    train_y = np.sinc(train_x[:,0] / np.pi) + 0.1 * train_x[0:, 1]
    train_y += np.matrix(np.random.normal(0, 0.1, size=train_y.shape))

    cl = rvm.RVMRegression(kernel="2D ex", gamma=1/10)
    cl.fit(train_x, train_y)
    rel_vec += len(cl.rel_ind)

    valid_x = np.matrix(np.random.uniform(-10, 10, size=(100, 2)))
    valid_y = np.sinc(valid_x[:, 0] / np.pi) + 0.1 * valid_x[0:, 1]

    pred_y = cl.predict(valid_x)

    RMSE += mean_squared_error(valid_y, pred_y) ** 0.5

print("Vectors: ", rel_vec / n_exp)
print("Root mean square = ", RMSE / n_exp)
