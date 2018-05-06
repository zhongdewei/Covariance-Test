from scipy.stats import ortho_group
import sklearn.linear_model as lm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
from statsmodels.graphics.gofplots import qqplot_2samples, qqplot

start_time = time.time()
n = 100
p = 100
t = []
rep_time = 5000
alpha1_list = []
alpha2_list = []
for i in range(rep_time):
    X = ortho_group.rvs(dim=n)

   # X = ortho_group.rvs(dim=n)
   # X = [a[:p] for a in X]
    y = np.random.normal(0,1,n)
    X = np.array(X)
    y = np.array(y)
    path = lm.lars_path(X,y)
    alpha2 = path[0][1]
    lam = lm.Lasso(alpha=alpha2, fit_intercept=False)
    lam = lam.fit(X,y)
    yhat = lam.predict(X)
    t.append(sum(yhat*y))
    alpha1_list.append(path[0][0])
    alpha2_list.append(alpha2)

print(np.mean(t))
print(np.std(t))
print("---%s seconds---"%(time.time()-start_time))
exp1 = np.random.exponential(0.4, 1000)
t = np.array(t)
plt.hist(t)
plt.show()
plt.clf()
stats.probplot(t, dist="expon", plot=plt)
plt.title("Normal Q-Q plot")
plt.show()
plt.clf()
plt.scatter(t, alpha1_list)
plt.savefig("alpha1.png")
plt.show()
plt.scatter(t, alpha2_list)
plt.savefig("alpha2.png")
plt.show()
plt.scatter(t, np.array(alpha1_list)-np.array(alpha2_list))
plt.savefig("alpha1-alpha2.png")
plt.show()
print("---%s seconds---"%(time.time()-start_time))
exit()


cdf_exponential = []
for i in range(1000):
    cdf_exponential.append(stats.expon.cdf(0.01*i))
cdf_exponential = np.array(cdf_exponential)


cdf_experiment = []
for i in range(1000):
    cnt = 0
    for e in t:
        if e <= 0.01*i:
            cnt += 1
    cdf_experiment.append(cnt/rep_time)
cdf_experiment = np.array(cdf_experiment)
x_axis = np.arange(0, 10, 0.01)
plt.plot(x_axis, cdf_experiment, label="experiment")
plt.plot(x_axis, cdf_exponential, label="exponential")
plt.legend(loc='lower right')
plt.savefig("100cdf_beta0.png")
plt.clf()
plt.plot(x_axis, cdf_experiment-cdf_exponential)
plt.xlabel("T_k")
plt.ylabel("cdf_experiment-cdf_exponential")
plt.savefig("100error_beta0.png")