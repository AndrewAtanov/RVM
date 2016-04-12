import numpy as np
from rvm import RVMClassification
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

data = np.loadtxt("data_sets/german/german.data-numeric.txt")

for el in data:
    if el[-1] == 2:
        el[-1] = 0

indexes = [i for i in range(24)]
data_x = data[:, indexes]
data_t = data[:, [24]]
data_t = data_t.reshape((len(data_t),))

train_x, test_x, train_y, test_y = train_test_split(data_x, data_t, test_size=0.3)

cl = RVMClassification(kernel='rbf')
cl.fit(train_x, train_y)

print(accuracy_score(cl.predict(test_x), test_y))
