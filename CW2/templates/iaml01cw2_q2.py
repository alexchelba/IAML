
##########################################################
#  Python script template for Question 2 (IAML Level 10)
#  Note that
#  - You should not change the name of this file, 'iaml01cw2_q2.py', which is the file name you should use when you submit your code for this question.
#  - You should write code for the functions defined below. Do not change their names.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define additional functions, do not define them here, but put them in a separate Python module file, "iaml01cw2_my_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission.
##########################################################

#--- Code for loading modules and the data set and pre-processing --->
# NB: You can edit the following and add code (e.g. code for loading sklearn) if necessary.

import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.decomposition as skd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, confusion_matrix,accuracy_score
import math

from iaml01cw2_my_helpers import *
import sys
sys.path.insert(1,'../helpers')
from iaml01cw2_helpers import *

#<----
Xtrn, Ytrn, Xtst, Ytst = load_FashionMNIST('../licenses/Fashion-MNIST')
Xtrn_orig = np.copy(Xtrn)
Xtst_orig = np.copy(Xtst)
Xtrn = np.true_divide(Xtrn,255.0) #Xtrn/255.0
Xtst = np.true_divide(Xtst,255.0) #Xtst/255.0
Xmean = Xtrn.mean(axis=0)
Xtrn_nm = np.subtract(Xtrn,Xmean)
Xtst_nm = np.subtract(Xtst,Xmean)

# Q2.1
def iaml01cw2_q2_1():
    log_reg = LogisticRegression()
    log_reg.fit(Xtrn_nm, Ytrn)
    X_new = log_reg.predict(Xtst_nm)
    print("Logistic Regression")
    print(accuracy_score(Ytst, X_new))
    print('\n')
    print(confusion_matrix(Ytst, X_new))

# iaml01cw2_q2_1()   # comment this out when you run the function

# Q2.2
def iaml01cw2_q2_2():
    svm = SVC()
    svm.fit(Xtrn_nm,Ytrn)
    X_new = svm.predict(Xtst_nm)
    print("SVM: ")
    print(accuracy_score(Ytst, X_new))
    print('\n')
    print(confusion_matrix(Ytst, X_new))

# iaml01cw2_q2_2()   # comment this out when you run the function

# Q2.3
def iaml01cw2_q2_3():
    pca = skd.PCA(n_components=2)
    PCs = pca.fit_transform(Xtrn_nm)
    log_reg = LogisticRegression()
    log_reg.fit(Xtrn_nm, Ytrn)
    std1 = PCs[:,0].std()
    std2 = PCs[:,1].std()
    x1 = np.linspace(-5*std1, 5*std1, 100)
    x2 = np.linspace(-5*std2, 5*std2, 100)
    xx,yy = np.meshgrid(x1,x2)
    grid = np.c_[xx.ravel(),yy.ravel()]
    gridf = pca.inverse_transform(grid)
    Z = log_reg.predict(gridf)
    Z = Z.reshape(xx.shape)
    lvls = [0,1,2,3,4,5,6,7,8,9]
    clrs = plt.cm.get_cmap('coolwarm')
    plt.contourf(xx, yy, Z, lvls, cmap = clrs)
    
    #plt.scatter(PCs[:, 0], PCs[:, 1],
     #       c=Ytrn, edgecolor='none', alpha=0.5,
      #      cmap=clrs)
      
    plt.colorbar()
    plt.title('Decision region for Logistic Regression Classifier')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()

# iaml01cw2_q2_3()   # comment this out when you run the function

# Q2.4
def iaml01cw2_q2_4():
    pca = skd.PCA(n_components=2)
    PCs = pca.fit_transform(Xtrn_nm)
    svm = SVC()
    svm.fit(Xtrn_nm,Ytrn)
    std1 = PCs[:,0].std()
    std2 = PCs[:,1].std()
    x1 = np.linspace(-5*std1, 5*std1, 100)
    x2 = np.linspace(-5*std2, 5*std2, 100)
    xx,yy = np.meshgrid(x1,x2)
    grid = np.c_[xx.ravel(),yy.ravel()]
    gridf = pca.inverse_transform(grid)
    Z = svm.predict(gridf)
    Z = Z.reshape(xx.shape)
    clrs = plt.cm.get_cmap('coolwarm')
    lvls = [0,1,2,3,4,5,6,7,8,9]
    plt.contourf(xx, yy, Z, lvls, cmap = clrs)
    
    #plt.scatter(PCs[:, 0], PCs[:, 1],
     #       c=Ytrn, edgecolor='none', alpha=0.5,
      #      cmap=clrs)
            
    plt.colorbar()
    plt.title('Decision region for SVM Classifier')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()

# iaml01cw2_q2_4()   # comment this out when you run the function

# Q2.5
def iaml01cw2_q2_5():
    Xsmall=[]
    Ysmall=[]
    counter = np.zeros(10)
    for i in range(len(Xtrn_nm)):
        if counter[Ytrn[i]]<1000:
            Xsmall.append(Xtrn_nm[i])
            Ysmall.append(Ytrn[i])
            counter[Ytrn[i]]+=1
    Xsmall = np.array(Xsmall)
    Ysmall = np.array(Ysmall)
    Cs = np.logspace(-2,3,10)
    svm = SVC(kernel='rbf', gamma='auto')
    means = []
    maxx = 0.
    Cmax = 0.
    for C in Cs:
        svm.C = C
        this_scores = cross_val_score(svm, Xsmall, Ysmall, cv=3, scoring='accuracy')
        m = this_scores.mean()
        means.append(m)
        if m>maxx:
            maxx = m
            Cmax = C
    plt.plot(Cs, means)
    plt.xlabel('(log-scale) Regularisation parameter')
    plt.ylabel('Mean accuracies')
    plt.axes().set_xscale('log')
    plt.title('Plot of mean accuracy for 10 evenly log-spaced values of C')
    print('Max mean accuracy = {} for C = {}'.format(round(maxx,5),round(Cmax,4)))
    plt.show()
    return Cmax

# iaml01cw2_q2_5()   # comment this out when you run the function

# Q2.6 
def iaml01cw2_q2_6():
    C = iaml01cw2_q2_5()
    svm = SVC(C=C, kernel='rbf',gamma='auto')
    svm.fit(Xtrn_nm,Ytrn)
    y_pred_trn = svm.predict(Xtrn_nm)
    y_pred_tst = svm.predict(Xtst_nm)
    acc_trn = accuracy_score(Ytrn,y_pred_trn)
    acc_tst = accuracy_score(Ytst,y_pred_tst)
    print("train set accuracy: {}".format(acc_trn))
    print("test set accuracy: {}".format(acc_tst))

# iaml01cw2_q2_6()   # comment this out when you run the function

