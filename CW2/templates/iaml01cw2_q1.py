
##########################################################
#  Python script template for Question 1 (IAML Level 10)
#  Note that
#  - You should not change the name of this file, 'iaml01cw2_q1.py', which is the file name you should use when you submit your code for this question.
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
from sklearn.metrics import mean_squared_error
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

# Q1.1
def iaml01cw2_q1_1():
    print(np.round(Xtrn_nm[0,:4],6))
    print(np.round(Xtrn_nm[-1,:4],6))

# iaml01cw2_q1_1()   # comment this out when you run the function

# Q1.2
def iaml01cw2_q1_2():
    fig,axes = plt.subplots(10,5, figsize = (10,10))
    for c in range(10):
        maxx = 0
        minn = 200000
        f_i1=np.array([])
        f_i2=np.array([])
        c_i1=np.array([])
        c_i2=np.array([])
        lbl_f_i1 = -1
        lbl_f_i2 = -1
        lbl_c_i1 = -1
        lbl_c_i2 = -1
        X_c = [Xtrn[i,:] for i in range(len(Xtrn)) if Ytrn[i]==c]
        lbl_i = [i for i in range(len(Xtrn)) if Ytrn[i]==c]
        X_c = np.array(X_c)
        Xmean_c = X_c.mean(axis=0)
        for i in range(len(X_c)):
            dist = np.linalg.norm(X_c[i]-Xmean_c)
            if dist>maxx:
                f_i2 = f_i1
                f_i1 = X_c[i]
                maxx=dist
                lbl_f_i2 = lbl_f_i1
                lbl_f_i1 = lbl_i[i]
            if dist<minn:
                c_i2 = c_i1
                c_i1 = X_c[i]
                minn = dist
                lbl_c_i2 = lbl_c_i1
                lbl_c_i1 = lbl_i[i]
        f_i1 = f_i1.reshape(28,28)
        f_i2 = f_i2.reshape(28,28)
        c_i1 = c_i1.reshape(28,28)
        c_i2 = c_i2.reshape(28,28)
        Xmean_c = Xmean_c.reshape(28,28)
        axes[c,0].set_title('Mean vector')
        axes[c,0].imshow(Xmean_c, cmap = plt.get_cmap('gray_r'))
        axes[c,1].set_title("Index: " + str(lbl_c_i1))
        axes[c,1].imshow(c_i1, cmap = plt.get_cmap('gray_r'))
        axes[c,2].set_title("Index: " + str(lbl_c_i2))
        axes[c,2].imshow(c_i2, cmap = plt.get_cmap('gray_r'))
        axes[c,3].set_title("Index: " + str(lbl_f_i2))
        axes[c,3].imshow(f_i2, cmap = plt.get_cmap('gray_r'))
        axes[c,4].set_title("Index: " + str(lbl_f_i1))
        axes[c,4].imshow(f_i1, cmap = plt.get_cmap('gray_r'))
    
    rows = ['Class {}'.format(row) for row in range(10)]
    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=90, size='large')

    fig.tight_layout()
    plt.show()
        
        
# iaml01cw2_q1_2()   # comment this out when you run the function

# Q1.3
def iaml01cw2_q1_3():
    pca = skd.PCA()
    pca.fit(Xtrn_nm)
    vars = np.array([pca.explained_variance_[:5]])
    vars = np.round(vars,4)
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    rows = ["PC " + str(i) for i in [1,2,3,4,5]]
    plt.table(cellText=vars.T, cellLoc='center', rowLabels=rows, colLabels=['Variance'], colWidths=[0.2] ,loc='center')
    plt.tight_layout()
    plt.show()

# iaml01cw2_q1_3()   # comment this out when you run the function


# Q1.4
def iaml01cw2_q1_4():
    pca = skd.PCA()
    pca.fit(Xtrn_nm)
    vars = np.array(pca.explained_variance_ratio_)
    cumvars = np.cumsum(vars)
    plt.plot(range(len(Xtrn_nm[0])), cumvars)
    plt.xlabel('Number of principal components')
    plt.ylabel('Cumulative explained variance ratio')
    plt.show()

# iaml01cw2_q1_4()   # comment this out when you run the function


# Q1.5
def iaml01cw2_q1_5():
    pca = skd.PCA()
    pca.fit(Xtrn_nm)
    fig = plt.figure(figsize=(16, 6))
    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        ax.set_title('PC ' + str(i+1))
        ax.imshow(pca.components_[i].reshape(28,28),
                cmap=plt.get_cmap('gray_r'))
    plt.show()

# iaml01cw2_q1_5()   # comment this out when you run the function


# Q1.6
def iaml01cw2_q1_6():
    table_content = []
    for i in range(10):
        X_i = [Xtrn_nm[j] for j in range(len(Xtrn_nm)) if Ytrn[j]==i]
        rmses = []
        for k in [5,20,50,200]:
            pca = skd.PCA(k)
            pca.fit(Xtrn_nm)
            X_pca = pca.transform([X_i[0]])
            X_new = pca.inverse_transform(X_pca)
            rmse = math.sqrt(mean_squared_error(X_i[0], X_new[0]))
            rmses.append(rmse.__round__(4))
        table_content.append(rmses)
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    rows = ["Class " + str(i) for i in range(10)]
    cols = ["K = " + str(j) for j in [5,20,50,200]]
    plt.table(cellText = table_content, cellLoc='center', rowLabels=rows, colLabels=cols, loc = 'center')
    plt.tight_layout()
    plt.show()

# iaml01cw2_q1_6()   # comment this out when you run the function


# Q1.7
def iaml01cw2_q1_7():
    fig, axes = plt.subplots(10,4, figsize = (10,10))
    for i in range(10):
        X_i = [Xtrn_nm[j] for j in range(len(Xtrn_nm)) if Ytrn[j]==i]
        for j,k in enumerate([5,20,50,200]):
            pca = skd.PCA(k)
            pca.fit(Xtrn_nm)
            X_pca = pca.transform([X_i[0]])
            X_new = pca.inverse_transform(X_pca)
            X_new +=Xmean
            X_new = X_new.reshape(28,28)
            axes[i,j].imshow(X_new, cmap = plt.get_cmap('gray_r'))
    rows = ['Class {}'.format(row) for row in range(10)]
    cols = ['K = {}'.format(col) for col in [5,20,50,200]]
    
    pad = 5 # in points

    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    
    fig.tight_layout()
    plt.show()

# iaml01cw2_q1_7()   # comment this out when you run the function


# Q1.8
def iaml01cw2_q1_8():
    pca = skd.PCA(2)
    X_new = pca.fit_transform(Xtrn_nm)
    plt.scatter(X_new[:, 0], X_new[:, 1],
            c=Ytrn, edgecolor='none',
            cmap=plt.cm.get_cmap('coolwarm', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()

# iaml01cw2_q1_8()   # comment this out when you run the function
