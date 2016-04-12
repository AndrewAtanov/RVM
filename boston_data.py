import numpy as np
from sklearn.metrics import mean_squared_error
from rvm import RVMRegression
from sklearn.datasets import load_boston

print('Boston Housing data set')
n_exp = 1
rel_vec = 0
RMSE = 0
for i in range(n_exp):
    boston = load_boston()
    train_x = boston.data[:481]
    train_y = boston.target[:481]

    valid_x = boston.data[481:]
    valid_y = boston.target[481:]

    cl = RVMRegression(kernel="rbf", gamma=0.0001)
    print(train_x.shape, train_y.shape)
    cl.fit(np.matrix(train_x), np.matrix(train_y.reshape(481, 1)))
    rel_vec += len(cl.rel_ind)

    pred_y = cl.predict(valid_x)

    RMSE += mean_squared_error(valid_y, pred_y) ** 0.5

print("Vectors: ", rel_vec / n_exp)
print("Root mean square = ", RMSE / n_exp)



