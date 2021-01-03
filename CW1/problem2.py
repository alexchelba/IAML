import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# a.
df = pd.read_csv('data/regression_part2.csv')
print("a.", df.info())
X = df["input"].values
y = df["output"].values

MSES = []

# M=1
lr = LinearRegression(fit_intercept=False)
X_b = np.c_[np.ones((X.shape[0],1)), X]
lr.fit(X_b,y)
y_pred = lr.predict(X_b)
plt.figure(figsize=(10,5))
plt.scatter(X,y,s=15)
plt.plot(X,y_pred,color='r', label = "degree 1")
plt.xlabel('Predictor', fontsize=16)
plt.ylabel('Target', fontsize=16)
MSES.append(mean_squared_error(y,y_pred))
# plt.show()

# M=2,3,4
inds = X.argsort()
X_sorted = X[inds]
y_sorted = y[inds]
X_op = X_sorted[:, np.newaxis]
colors = ['teal', 'yellowgreen', 'gold']
for count, degree in enumerate([2,3,4]):    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_op)
    lin = LinearRegression(fit_intercept=False)
    lin.fit(X_poly,y_sorted)
    y_plot = lin.predict(X_poly)
    plt.plot(X_sorted,y_plot, color = colors[count], label = "degree %d" % degree)
    MSES.append(mean_squared_error(y_plot, y_sorted))

plt.legend()
plt.show()
plt.clf()

# b.
print("MSEs for polynomial m=1,2,3,4: ", MSES)
inds = [1,2,3,4]
plt.bar(inds, MSES)
plt.xlabel("Degree of polynomial")
plt.ylabel("MSE")
plt.show()
plt.clf()

# c. M=3 MSE=2.78, M=4 MSE=2.74

# d.
def rbf(x,c,l):
    res = []
    for r in x:
        x0 = [1]
        for i in c:
            x0.append(math.exp(-(((r[0]-i)*(r[0]-i))/(2*l*l))))
        res.append(x0)
    return res

plt.scatter(X,y,s=15)
lengths = [0.2, 100, 1000]
c = [-4.0, -2.0, 2.0, 4.0]
for count, alpha in enumerate(lengths):
    X_rbf = rbf(X_op, c, alpha)
    lin = LinearRegression(fit_intercept=False)
    lin.fit(X_rbf, y_sorted)
    y_plot = lin.predict(X_rbf)
    plt.plot(X_sorted,y_plot, color = colors[count], label = "length %f" % alpha)
plt.xlabel("Predictor")
plt.ylabel("Target")
plt.legend()
plt.show()


