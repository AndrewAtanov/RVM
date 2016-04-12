from rvm import RVMClassification
import numpy as np
import matplotlib.pyplot as plt
# import warnings

# warnings.filterwarnings('error')

np.random.seed(2)

n_ex = 50 * 2

train_x = []
train_y = []
for i in range(n_ex // 2):
    p = np.random.rand(1, 2)[0, :]
    p[0] *= np.pi
    p[1] *= 1.5
    while np.sin(p[0]) < p[1]:
        p = np.random.rand(1, 2)[0, :]
        p[0] *= np.pi
        p[1] *= 1.5

    train_x.append(p)
    train_y.append(1)

for i in range(n_ex // 2):
    p = np.random.rand(1, 2)[0, :]
    p[0] *= np.pi
    p[1] *= 1.5
    while np.sin(p[0]) > p[1]:
        p = np.random.rand(1, 2)[0, :]
        p[0] *= np.pi
        p[1] *= 1.5
    train_x.append(p)
    train_y.append(0)

train_x = np.matrix(train_x)
train_y = np.array(train_y)

cl = RVMClassification(kernel="rbf", gamma=1)
cl.fit(train_x, train_y)

valid_x = np.matrix(np.random.uniform(0, 1, size=(50, 2)))
valid_x[:, 1] *= 1.5
valid_x[:, 0] *= np.pi

valid_y = np.sign(np.sin(valid_x[:, 0]) - valid_x[:, 1])

valid_y[valid_y == -1] = 0

predicted_y = cl.predict(valid_x)

plt.plot(train_x[:, 0].A[:n_ex // 2], train_x[:, 1].A[:n_ex // 2], 'rx')
plt.plot(train_x[:, 0].A[n_ex // 2:], train_x[:, 1].A[n_ex // 2:], 'gx')

green_x = []
red_x = []

err = 0

for i in range(len(predicted_y)):
    if predicted_y[i] == 1:
        red_x.append([valid_x.A[i, 0], valid_x.A[i, 1]])
        if valid_y[i] == -1:
            err += 1
    else:
        green_x.append([valid_x.A[i, 0], valid_x.A[i, 1]])
        if valid_y[i] == 1:
            err += 1


green_x = np.array(green_x)
red_x = np.array(red_x)

print(err)

plt.plot(red_x[:, 0], red_x[:, 1], 'ro')
plt.plot(green_x[:, 0], green_x[:, 1], 'go')

x = np.arange(0, np.pi, 0.1)
y = np.sin(x)

plt.plot(x, y, 'y')

plt.show()
