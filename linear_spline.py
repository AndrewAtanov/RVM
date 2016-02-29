import numpy as np
import matplotlib.pyplot as plt
import rvm

L = -10
R = 10
train_x = np.arange(L, R, (R - L) / 100)
train_y = np.sinc(train_x / np.pi)

train_y += np.random.uniform(-0.2, 0.2, len(train_y))
train_x = np.array([[x] for x in train_x])
train_y = np.array([[x] for x in train_y])

valid_x = np.array([[x] for x in np.arange(L, R, 0.05)])

cl = rvm.RVMRegression(kernel='linear spline')
cl.fit(train_x, train_y)

valid_y = cl.predict(valid_x)

relevance_x = []
relevance_y = []
relevance_w = []

for i in range(len(train_x)):
    if abs(cl.w[i]) > 0.001:
        relevance_x.append(train_x[i])
        relevance_y.append(train_y[i])
        relevance_w.append(cl.w[i])
print(len(relevance_w))
plt.plot(train_x, train_y, "ro")

plt.plot(valid_x, valid_y)
plt.plot(relevance_x, relevance_y, "go")


x = np.arange(L, R, 0.2)
y = np.sinc(x / np.pi)

plt.plot(x, y, "y")

plt.show()
