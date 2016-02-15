__author__ = 'andrew'
import numpy as np


class RVMRegression:
    """RVM Regression class"""
    fitted = False
    alpha = None
    betta = None
    kernel = 'linear'
    data_x = None
    data_t = None
    alpha_bound = 10**12
    weight_bound = 10**-6
    number_of_iterations = 100
    F = None
    kernel_m = None
    D = 0
    w = None
    inv_a = None
    N = 0

    def __init__(self, alpha=None, betta=None, kernel='linear', coef0=0, gamma=1, degree=2, number_of_iterations=100):
        if alpha:
            self.alpha = alpha

        if betta:
            self.betta = betta

        self.coef0 = coef0
        self.degree = degree
        self.gamma = gamma
        self.kernel = kernel
        self.number_of_iterations = number_of_iterations

    def fit(self, x, t):
        if len(x) == 0:
            raise NameError("X array is empty!")
        if len(x) != len(t):
            raise NameError("Different len X and t")

        self.D = len(x[0])
        self.N = len(x)

        if self.kernel in ['linear']:
            self.kernel_m = len(x[0])

        if self.kernel in ['rbf', 'poly']:
            self.kernel_m = len(x) + 1

        self.data_x = x.copy()
        self.data_t = t.copy()

        if not self.alpha:
            self.alpha = np.array([1.] * self.kernel_m)

        if not self.betta:
            self.betta = 1

        self.create_f()
        gamma = np.array([0.] * self.kernel_m)
        w_mp = np.matrix([[]])
        t_matrix = np.matrix(self.data_t)

        print(self.validity())

        for k in range(self.number_of_iterations):
            sigma = np.linalg.inv(self.betta * (self.F.T * self.F) + np.matrix(np.diag(self.alpha)))
            w_mp = sigma * self.F.T * (self.betta * t_matrix)
            for j in range(self.kernel_m):
                if np.abs(w_mp.A[j]) < self.weight_bound or np.abs(self.alpha[j]) > self.alpha_bound:
                    w_mp.A[j] = 0
                    self.alpha[j] = 1e200
                    gamma[j] = 0
                else:
                    gamma[j] = 1 - self.alpha[j] * sigma.A[j][j]
                    self.alpha[j] = gamma[j] / (w_mp.A[j][0] ** 2)
            if (np.linalg.norm(t_matrix - self.F * w_mp) ** 2) == 0:
                self.betta = 1e100
                break
            else:
                self.betta = (self.N - sum(gamma)) / (np.linalg.norm(t_matrix - self.F * np.matrix(w_mp)) ** 2)

        self.w = w_mp

    def create_f(self):
        self.F = []
        for i in range(len(self.data_t)):
            self.F.append([self.phi(self.data_x[i], j) for j in range(self.kernel_m)])
        self.F = np.matrix(self.F)

    def calc_sigma(self):
        a = np.matrix(np.diag(self.alpha))
        try:
            inv_a = np.linalg.inv(a)
        except np.linalg.LinAlgError:
            inv_a = np.linalg.pinv(a)

        self.inv_a = inv_a.copy()
        e = np.matrix(np.diag([1] * a.shape[0]))
        tmp_mat = np.linalg.pinv(e + self.betta * self.F * self.inv_a * self.F.T)
        return self.inv_a - self.inv_a * self.betta * self.F.T * tmp_mat * self.F * self.inv_a

    def get_m(self):
        return self.kernel_m
        if self.kernel in ['linear', 'poly']:
            return len(self.data_x[0])

    def phi(self, x, i):
        if self.kernel == 'linear':
            return x[i]
        if self.kernel == 'poly':
            if i == self.kernel_m - 1:
                return 1
            res = self.coef0
            for j in range(len(x)):
                res += x[j] * self.data_x[i][j]
            return res ** self.degree
        if self.kernel == 'rbf':
            if i == self.kernel_m - 1:
                return 1
            return np.exp((-1.) * self.gamma * (np.linalg.norm(x - self.data_x[i]) ** 2))

    def predict(self, data_x):
        if self.w is None:
            raise NameError("Classifier don't fitted yet!")
        t_ans = []
        for x in data_x:
            t = 0
            for j in range(self.kernel_m):
                t += self.w.A[j] * self.phi(x, j)
            t_ans.append(t)
        return np.array(t_ans)

    def validity(self, x=None):
        if x is not None:
            t = self.predict(x)
        else:
            t = self.data_t

        self.inv_a = np.linalg.inv(np.matrix(np.diag(self.alpha)))
        tmp = self.F * self.inv_a * self.F.T
        e = np.matrix(np.diag([1 / self.betta] * tmp.shape[0]))

        u = e + tmp
        tmp = np.linalg.inv(u)
        tmp = -0.5 * (t.T * tmp * t)
        det = np.linalg.det(u)
        tmp = tmp.A[0][0] - np.log(2 * np.pi)*(self.N / 2) - np.log(det) / 2
        return tmp
