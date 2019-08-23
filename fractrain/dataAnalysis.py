import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.special import expit

for i in range(1,5):
    for j in range(1,4):
        print(i,j)
        # if i == 3 and j == 2: continue
        with open("condition"+str(i)+"/log"+str(j)) as dat:
            log1 = [(0 if ("Inco" in ln) else 1) for ln in dat]

        mod = linear_model.LogisticRegression(C=1e5, solver='lbfgs')
        mod.fit(np.asarray([i for i in range(len(log1))]).reshape(-1, 1),np.asarray(log1).reshape(-1,1))

        plt.figure(1, figsize=(4, 3))
        plt.clf()
        plt.scatter(np.asarray([i for i in range(len(log1))]).reshape(-1, 1),np.asarray(log1).reshape(-1,1))
        X_test = np.linspace(-100, 1300, 3000)
        loss = expit(X_test * mod.coef_ + mod.intercept_).ravel()
        plt.plot(X_test, loss, color="red")

        plt.show()
