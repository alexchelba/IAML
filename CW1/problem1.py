import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error

# a.
df = pd.read_csv('data/regression_part1.csv')
print("a.", df.describe())
X = df["revision_time"].values
y = df["exam_score"].values

# b.
X_train_b = np.c_[np.ones((X.shape[0],1)), X]
lr = LinearRegression(fit_intercept=False)
lr.fit(X_train_b, y)
exam_predict = lr.predict(X_train_b)
#print(exam_predict)
print(lr.coef_)

# c.
ax1 = df.plot(kind='scatter', x='revision_time', y='exam_score', color='blue', alpha=0.5, figsize=(10, 7))
plt.plot(X, exam_predict)
plt.show()
plt.clf()

# d.
# Calculate the mean of X and y
xmean = np.mean(X)
ymean = np.mean(y)

# Calculate the terms needed for the numator and denominator of beta
xycov = (X - xmean) * (y - ymean)
xvar = (X - xmean)**2

# Calculate beta and alpha
beta = xycov.sum() / xvar.sum()
alpha = ymean - (beta * xmean)
y_pred = alpha + beta * X
plt.plot(X, y_pred)
plt.show()
plt.clf()

# e. MSE = 1/n * sum(i=1,n)[(Y(i)-Y_hat(i))^2]
# mean squared error has the disadvantage of heavily weighting outliers.
# This is a result of the squaring of each term, which effectively weights large errors more heavily than small ones.

# f.
print("sklearn MSE: ", mean_squared_error(exam_predict,y))
print("Manual MSE: ", mean_squared_error(y_pred, y))

# g.
w0 = 20.0
w1 = np.linspace(-2,2,100)
wmin = 200000000
minn=200000000
for w1e in w1:
    y_pred_t = w0 + w1e * X
    #plt.plot(X, y_pred_t)
    mse = mean_squared_error(y_pred_t, y)
    plt.scatter(w1e,mse)
    if mse<=minn:
        minn=mse
        wmin=w1e

plt.xlabel('Gradient')
plt.ylabel('MSE')
plt.show()
print("Minimum MSE for w0=20: ", minn, "at: ", wmin)