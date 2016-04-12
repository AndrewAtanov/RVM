
import rvm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

print('2D sinc regression example')

np.random.seed(1)

train_x = np.matrix(np.random.uniform(-10, 10, size=(100, 2)))
train_y = np.sinc(train_x[:,0] / np.pi) + 0.1 * train_x[0:,1]
train_y += np.matrix(np.random.normal(0, 0.1, size=train_y.shape))
#train_y = train_x[:, 0]


X, Y = np.meshgrid(np.arange(-10, 10, 0.5), np.arange(-10, 10, 0.5))
Z = np.sinc(X / np.pi) + 0.1 * Y
#Z = X

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)


ax.scatter(train_x.A[:, 0], train_x.A[:, 1], train_y.A[:, 0], marker='o', c='r')

cl = rvm.RVMRegression(kernel="2D ex", gamma=1/10)
cl.fit(train_x, train_y)


X, Y = np.meshgrid(np.linspace(-10, 10, 60), np.linspace(-10, 10, 60))
valid_x = []
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        valid_x.append([X[i][j], Y[i][j]])
valid_x = np.matrix(valid_x)

valid_y_ = cl.predict(valid_x)
valid_y = valid_y_.reshape(X.shape)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_wireframe(X, Y, valid_y, rstride=2, cstride=2)
ax.scatter(train_x.A[:, 0], train_x.A[:, 1], train_y.A[:, 0], marker='o', c='r')

print("RELEVANT", len(cl.rel_ind))
# print(cl.validity())

RMS = np.linalg.norm(valid_y.reshape(valid_y.shape[0] ** 2) - (np.sinc(valid_x[:, 0] / np.pi) + 0.1 * valid_x[:, 1]))
RMSE = mean_squared_error(valid_y_, (np.sinc(valid_x.A[:, 0] / np.pi) + 0.1 * valid_x.A[:, 1])) ** 0.5
print("RMSE", RMSE)

print(cl.w)

plt.show()
