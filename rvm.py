from IPython.core.pylabtools import select_figure_formats

__author__ = 'andrew'

import numpy as np
from scipy.special import expit
from scipy.linalg import logm, expm


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
    eps = 1e-3
    lokelihood_eps = 1e-4

    def __init__(self, learning_rate=0.1, alpha=None, betta=None, kernel='linear', coef0=0, gamma=1, degree=2, number_of_iterations=100):
        if alpha:
            self.alpha = alpha

        if betta:
            self.betta = betta

        self.coef0 = coef0
        self.degree = degree
        self.gamma = gamma
        self.kernel = kernel
        self.number_of_iterations = number_of_iterations
        self.relevance_w = []
        self.relevance_x = []
        self.relevance_y = []
        self.learning_rate = learning_rate

    def tipping_re_estimation(self, n_iterations):
        gamma = np.array([0.] * self.kernel_m)
        w_mp = np.matrix([[]])
        t_matrix = np.matrix(self.data_t)

        last = self.marginal_likelihood()
        cur = last
        print("FIRST = ", last)
        iterations_number = 0

        for k in range(n_iterations):
        # while np.abs(cur - last) > self.lokelihood_eps:
            last = cur
            iterations_number += 1

            tmp_m = self.betta * (self.F.T * self.F) + np.matrix(np.diag(self.alpha))
            sigma = np.linalg.inv(self.betta * (self.F.T * self.F) + np.matrix(np.diag(self.alpha)))
            w_mp = sigma * self.F.T * (self.betta * t_matrix)

            alpha_old = self.alpha.copy()
            for j in range(self.kernel_m):
                if np.abs(w_mp.A[j]) < self.weight_bound or np.abs(self.alpha[j]) > self.alpha_bound:
                    w_mp.A[j] = 0
                    self.alpha[j] = 1e200
                    gamma[j] = 0
                    # gamma[j] = 1 - self.alpha[j] * sigma.A[j][j]
                else:
                    gamma[j] = 1 - self.alpha[j] * sigma.A[j][j]
                    self.alpha[j] = gamma[j] / (w_mp.A[j][0] ** 2)

            self.betta = (self.N - sum(gamma)) / (np.linalg.norm(t_matrix - self.F * np.matrix(w_mp)) ** 2)
            cur = self.marginal_likelihood()
            print(cur - last)

        print(iterations_number, "\nLIKLIHOOD", cur)

        self.w = w_mp

        return 1

    def EM_re_estimation(self, n_iterations, likelihood):
        gamma = np.array([0.] * self.kernel_m)
        w_mp = np.matrix([[]])
        t_matrix = np.matrix(self.data_t)

        last = self.marginal_likelihood()
        cur = last
        print("FIRST LIKELIHOOD = ", last)
        iterations_number = 0

        # for k in range(n_iterations):
        while cur < likelihood and iterations_number < n_iterations:
            last = cur
            iterations_number += 1

            tmp_m = self.betta * (self.F.T * self.F) + np.matrix(np.diag(self.alpha))
            sigma = np.linalg.inv(self.betta * (self.F.T * self.F) + np.matrix(np.diag(self.alpha)))
            w_mp = sigma * self.F.T * (self.betta * t_matrix)
            for j in range(self.kernel_m):
                if np.abs(w_mp.A[j]) < self.weight_bound or np.abs(self.alpha[j]) > self.alpha_bound:
                    w_mp.A[j] = 0
                    self.alpha[j] = 1e200
                    gamma[j] = 0
                else:
                    gamma[j] = 1 - self.alpha[j] * sigma.A[j][j]
                    self.alpha[j] = 1 / (w_mp.A[j][0] ** 2 + sigma.A[j][j])

            self.betta = (self.N - sum(gamma)) / (np.linalg.norm(t_matrix - self.F * np.matrix(w_mp)) ** 2)

            cur = self.marginal_likelihood()
            print(cur - last, cur, iterations_number)

        print(iterations_number, "\nLIKELIHOOD", last)
        self.w = w_mp

        return 1

    def fit(self, x, t):
        if len(x) == 0:
            raise NameError("X array is empty!")
        if len(x) != len(t):
            raise NameError("Different len X and t")

        self.D = len(x[0])
        self.N = len(x)

        self.kernel_m = self.N + 1

        if self.kernel in ['linear']:
            self.kernel_m = len(x.A[0])

        if self.kernel in ['rbf', 'poly']:
            self.kernel_m = len(x) + 1

        if self.kernel == '2D ex':
            self.kernel_m = len(x) + 4

        self.data_x = x.copy()
        self.data_t = t.copy()

        if not self.alpha:
            self.alpha = np.array([1.] * self.kernel_m)

        if not self.betta:
            self.betta = 1

        self.create_f()

        # self.tipping_re_estimation(self.number_of_iterations)
        self.EM_re_estimation(self.number_of_iterations, 69.2969836674)

        self.rel_ind = []

        for i in range(len(self.w)):
            if abs(self.w.A[i][0]) > self.eps:
                self.rel_ind.append(i)
                # self.relevance_w.append(self.w.A[i])
                # self.relevance_x.append(self.data_x[i] if i != self.kernel_m - 1 else [1])
                # self.relevance_y.append(self.data_t[i] if i != self.kernel_m - 1 else [0])

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

    def phi(self, x, i):

        if self.kernel == 'linear':
            return x.A[0][i]

        if self.kernel == 'poly':
            if i == self.kernel_m - 1:
                return 1
            # res = self.coef0
            res = x * self.data_x[i].T + self.coef0
            res **= self.degree
            return res.A[0][0]

        if self.kernel == 'rbf':
            if i == self.kernel_m - 1:
                return 1
            return np.exp((-1.) * self.gamma * (np.linalg.norm(x - self.data_x[i]) ** 2))

        if self.kernel == 'linear spline':
            if i == self.kernel_m - 1:
                return 1
            x_n = x[0]
            x_m = self.data_x[i][0]
            m = min(x_n, x_m)
            return 1 + x_n * x_m + x_n * x_m * m - (x_n + x_m) * (m**2) / 2 + m**3 / 3

        if self.kernel == '2D ex':
            if i == self.kernel_m - 1:
                return 1
            elif i == 0:
                return x.A[0][i]
            elif i == 1:
                return x.A[0][i]
            elif i == 2:
                return x.A[0][1] * x.A[0][0]
            return np.exp((-1.) * self.gamma * (np.linalg.norm(x.A[0][0] - self.data_x.A[i - 3][0]) ** 2) - 0.001 * (np.linalg.norm(x.A[0][1] - self.data_x.A[i - 3][1]) ** 2))

    def predict(self, data_x):
        if self.w is None:
            raise NameError("Classifier don't fitted yet!")
        t_ans = []
        for x in data_x:
            t = 0
            for j in self.rel_ind:
                t += self.w.A[j][0] * self.phi(x, j)
            t_ans.append(t)
        return np.array(t_ans)

    def marginal_likelihood(self, x=None):
        if x is not None:
            t = self.predict(x)
        else:
            t = self.data_t

        self.inv_a = np.matrix(np.diag(1 / self.alpha))
        tmp = self.F * self.inv_a * self.F.T
        e = np.matrix(np.diag([1 / self.betta] * tmp.shape[0]))

        u = e + tmp
        tmp = np.linalg.inv(u)
        tmp = -0.5 * (t.T * tmp * t)
        det = logm(u).trace()
        tmp = tmp.A[0][0] - np.log(2 * np.pi)*(self.N / 2) - det / 2
        return tmp


class RVMClassification(RVMRegression):
    """RVM Classification class"""

    def sigmoid(self, a):
        return 1 / (1 + np.exp(a))

    def calc_logproba(self, _w, a):
        ans = []
        if _w.shape == (self.N, 1):
            _w = [_w]
        for w in _w:
            t = self.data_t
            logprob = 0
            for i in range(len(t)):
                a_exp = -(self.F[i, :] * w).A[0][0]
                if t[i] == 1:
                    logprob += -np.log(1 + np.exp(a_exp))
                else:
                    logprob += a_exp - np.log(1 + np.exp(a_exp))
            logprob -= 0.5 * (w.T * a * w).A[0][0]
            ans.append(-logprob)
        return ans[0] if len(ans) == 1 else np.array(ans)

    def tipping_re_estimation(self, n_iterations):
        gamma = np.array([0.] * self.kernel_m)
        t_matrix = np.matrix(self.data_t)
        w_mp = t_matrix.T #np.array([[0.]] * self.kernel_m)

        cur = self.calc_logproba(w_mp, np.matrix(np.diag(self.alpha)))
        print("FIRST LIKELIHOOD = ", cur)
        iterations_number = 0

        for k in range(n_iterations):
        # while np.abs(cur - last) > self.lokelihood_eps:
            last = cur
            iterations_number += 1

            # tmp_m = self.betta * (self.F.T * self.F) + np.matrix(np.diag(self.alpha))
            # sigma = np.linalg.inv(self.betta * (self.F.T * self.F) + np.matrix(np.diag(self.alpha)))
            # w_mp = sigma * self.F.T * (self.betta * t_matrix)

            w_mp, sigma = self.w_mp_evaluating(t_matrix, w_mp)
            for j in range(self.kernel_m):
                if np.abs(w_mp.A[j]) < self.weight_bound or np.abs(self.alpha[j]) > self.alpha_bound:
                    w_mp.A[j] = 0
                    self.alpha[j] = 1e200
                    gamma[j] = 0
                    # gamma[j] = 1 - self.alpha[j] * sigma.A[j][j]
                else:
                    gamma[j] = 1 - self.alpha[j] * sigma.A[j][j]
                    self.alpha[j] = gamma[j] / (w_mp.A[j][0] ** 2)

            cur = self.calc_logproba(w_mp, np.matrix(np.diag(self.alpha)))
            print(cur - last)

        print(iterations_number, "\nLIKELIHOOD", cur)

        self.w = w_mp.copy()

        return 1

    def EM_re_estimation(self, n_iterations, likelihood=float('inf')):
        gamma = np.array([0.] * self.kernel_m)
        w_mp = np.matrix([[1.]] * self.kernel_m)
        t_matrix = np.matrix(self.data_t)

        cur = self.calc_logproba(w_mp, np.matrix(np.diag(self.alpha)))
        print("FIRST LIKELIHOOD = ", cur)
        iterations_number = 0

        while cur < likelihood and iterations_number < n_iterations:
            last = cur
            iterations_number += 1

            # tmp_m = self.betta * (self.F.T * self.F) + np.matrix(np.diag(self.alpha))
            # sigma = np.linalg.inv(self.betta * (self.F.T * self.F) + np.matrix(np.diag(self.alpha)))
            # w_mp = sigma * self.F.T * (self.betta * t_matrix)

            w_mp, sigma = self.w_mp_evaluating(t_matrix, w_mp)

            for j in range(self.kernel_m):
                if np.abs(w_mp.A[j]) < self.weight_bound or np.abs(self.alpha[j]) > self.alpha_bound:
                    w_mp.A[j] = 0
                    self.alpha[j] = 1e200
                    gamma[j] = 0
                else:
                    gamma[j] = 1 - self.alpha[j] * sigma.A[j][j]
                    self.alpha[j] = 1 / (w_mp.A[j][0] ** 2 + sigma.A[j][j])

            self.betta = (self.N - sum(gamma)) / (np.linalg.norm(t_matrix - self.F * np.matrix(w_mp)) ** 2)

            cur = self.calc_logproba(w_mp, np.matrix(np.diag(self.alpha)))
            print(cur - last, cur, iterations_number)

        print(iterations_number, "\nLIKELIHOOD", cur)
        self.w = w_mp

        return 1

    def w_mp_evaluating(self, t_matrix, w_mp_c):
        print('\nW_mp Evaluating')
        a = np.matrix(np.diag(self.alpha))
        bound = 1e-6

        sigma = np.matrix(np.zeros(a.shape))
        cur = self.calc_logproba(w_mp_c, a)
        last = 0
        print('LIKELIHOOD', cur)

        n_iteratoin = 0

        while np.abs(cur - last) > bound and n_iteratoin < 99: # np.abs(cur - last) > bound: # np.linalg.norm(w_mp_c - w_mp_old) > bound:
            n_iteratoin += 1
            last = cur
            s = []
            for i in range(self.N):
                s.append(1 / (1 + np.exp(-(self.F[i, :] * w_mp_c).A[0][0])))

            s = np.array(s)
            r = np.matrix(np.diag(s * (1 - s)))
            try:
                sigma = np.linalg.inv(self.F.T * r * self.F + a)
            except:
                print(self.alpha)

            w_mp_c = w_mp_c - self.learning_rate * sigma * (self.F.T * (np.matrix(s) - t_matrix).T + a * w_mp_c)
#             w_mp_c = sigma * F.T * r * t_matrix.T

            cur = self.calc_logproba(w_mp_c, a)

            s = []
            for i in range(self.N):
                s.append(1 / (1 + np.exp(-(self.F[i, :] * w_mp_c).A[0][0])))

            s = np.array(s)

            # print(cur - last, np.linalg.norm((self.F.T * (np.matrix(s) - t_matrix).T + a * w_mp_c)), cur)

        print('LIKELIHOOD', cur)
        return w_mp_c, sigma

    def fit(self, x, t):
        if len(x) == 0:
            raise NameError("X array is empty!")
        if len(x) != len(t):
            raise NameError("Different len X and t")

        self.D = len(x[0])
        self.N = len(x)

        self.kernel_m = self.N

        if self.kernel == "linear":
            self.kernel_m = x.shape[1] - 1

        self.data_x = x.copy()
        self.data_t = t.copy()

        if not self.alpha:
            self.alpha = np.array([1.] * self.kernel_m)

        if not self.betta:
            self.betta = 1

        self.create_f()
        gamma = np.array([0.] * self.kernel_m)
        t_matrix = np.matrix(self.data_t)
        w_mp_c = t_matrix.copy().reshape((len(self.data_t), 1))

        last = 0

        self.tipping_re_estimation(self.number_of_iterations)
        # self.EM_re_estimation(self.number_of_iterations)

    def predict(self, data_x):
        if self.w is None:
            raise NameError("Classifier did not fit yet!")
        t_ans = []
        for x in data_x:
            t = 0
            for j in range(self.kernel_m):
                t += self.w.A[j][0] * self.phi(x, j)
            t_ans.append(np.sign(t))
        return np.array([1 if x == 1 else 0 for x in t_ans])

    def phi(self, x, i):

        if self.kernel == 'linear':
            return x.A[0][i]

        if self.kernel == 'poly':
            res = x * self.data_x[i].T + self.coef0
            res **= self.degree
            return res.A[0][0]

        if self.kernel == 'rbf':
            return np.exp((-1.) * self.gamma * (np.linalg.norm(x - self.data_x[i]) ** 2))
