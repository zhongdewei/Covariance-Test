from scipy.stats import ortho_group
import sklearn.linear_model as lm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
import csv


start_time = time.time()
n = 100
p = 18
p1 = 1
rep_time = 2000
T_k = []
for i in range(rep_time):
    X_list = ortho_group.rvs(dim=n)
    X_list = [a[:p] for a in X_list]
    beta = np.array([50] * p1 + [0] * (p - p1))
    epsilon = np.random.normal(0, 1, n)
    X = np.array(X_list)
    y = np.matmul(X, beta) + epsilon
    path = lm.lars_path(X, y)
    alpha = path[0]
    index = path[1]
    beta = np.array([[50]] * p1 + [[0]] * (p - p1))
    epsilon = epsilon.reshape(n, 1)
    XA = [b[index[:p1]] for b in X_list]
    XA = np.array(XA)
    XA = XA.reshape(n, p1)
    alpha2 = alpha[p1+1]
    lam = lm.Lasso(alpha=alpha2, fit_intercept=False)
    lam = lam.fit(X, y)
    y_hat = lam.predict(X)
    lam_A = lm.Lasso(alpha=alpha2, fit_intercept=False)
    lam_A = lam_A.fit(XA, y)
    y_tilde = lam_A.predict(XA)
    y_tilde = y_tilde.reshape(1, n)
    T_k.append(np.sum(y * (y_hat - y_tilde)))

print("The mean is:", np.mean(T_k))
print("The std is: ", np.std(T_k))
print("---%s seconds"%(time.time()-start_time))
plt.hist(T_k)
plt.savefig("hist_p1=45.png")
plt.show()
stats.probplot(T_k, dist="expon", plot=plt)
plt.title("Normal Q-Q plot")
plt.savefig("qqplot_p1=45.png")
plt.show()

cdf_exponential = []
for i in range(1000):
    cdf_exponential.append(stats.expon.cdf(0.01*i))
cdf_exponential = np.array(cdf_exponential)


cdf_experiment = []
for i in range(1000):
    cnt = 0
    for e in T_k:
        if e <= 0.01*i:
            cnt += 1
    cdf_experiment.append(cnt/rep_time)
cdf_experiment = np.array(cdf_experiment)
x_axis = np.arange(0, 10, 0.01)
plt.plot(x_axis, cdf_experiment, label="experiment")
plt.plot(x_axis, cdf_exponential, label="exponential")
plt.legend(loc='lower right')
plt.savefig("cdf_p1=45.png")
plt.show()

plt.plot(x_axis, cdf_experiment-cdf_exponential)
plt.xlabel("T_k")
plt.ylabel("cdf_experiment-cdf_exponential")
plt.savefig("error_p1=45.png")
plt.show()