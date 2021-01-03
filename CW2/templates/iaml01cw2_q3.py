
##########################################################
#  Python script template for Question 3 (IAML Level 10)
#  Note that:
#  - You should not change the name of this file, 'iaml01cw2_q3.py', which is the file name you should use when you submit your code for this question.
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
import sklearn.cluster as skc
import sklearn.decomposition as skd
from sklearn.metrics import mean_squared_error
from sklearn import mixture
import math
from collections import Counter
from scipy.cluster.hierarchy import dendrogram, linkage

from iaml01cw2_my_helpers import *
import sys
sys.path.insert(1,'../helpers')
from iaml01cw2_helpers import *

#<----
Xtrn, Ytrn, Xtst, Ytst = load_CoVoST2('../data')
data =" "
ll = []
abbvs = []
with open('../data/languages.txt', 'r') as file:
    data = file.read().split('\n')
    for row in data:
        col = row.split()
        if(len(col)):
            ll.append(col[1])
            p = len(col[2])-1
            abbvs.append(col[2][1:p])

# Q3.1
def iaml01cw2_q3_1():
    kmeans = skc.KMeans(n_clusters=22, random_state=1)
    kmeans.fit(Xtrn, Ytrn)
    y_pred = kmeans.predict(Xtrn)
    cc = kmeans.cluster_centers_
    sum=0
    for i in range(len(Xtrn)):
        dist = np.linalg.norm(Xtrn[i]-cc[y_pred[i]])
        sum+=dist**2
    print(sum)
    print(sorted(Counter(y_pred).items()))
    return cc

# iaml01cw2_q3_1()   # comment this out when you run the function

# Q3.2
def iaml01cw2_q3_2():
    mean_v = np.zeros([22,26])
    count = np.zeros([22,1])
    for i in range(len(Xtrn)):
        mean_v[Ytrn[i]]+=Xtrn[i]
        count[Ytrn[i]]+=1
    mean_v = np.true_divide(mean_v,count)
    pca = skd.PCA(2)
    pca.fit(mean_v)
    PCs = pca.transform(mean_v)
    lvls = np.arange(22)
    plt.scatter(PCs[:,0], PCs[:,1], s=50, label = 'Mean vectors', c= lvls, cmap=plt.cm.get_cmap('coolwarm'))
    cc = iaml01cw2_q3_1()
    pccs =pca.transform(cc)
    plt.scatter(pccs[:,0], pccs[:,1], marker = '*', label = 'Cluster centers', c=lvls, cmap=plt.cm.get_cmap('coolwarm', 22))
    for i, txt in enumerate(abbvs):
        plt.annotate(txt, (PCs[i,0]+.003, PCs[i,1]+.003))
    plt.legend()
    plt.title('Plot for Mean vector and cluster centre of each language')
    plt.colorbar(ticks=lvls)
    plt.show()
    

# iaml01cw2_q3_2()   # comment this out when you run the function

# Q3.3
def iaml01cw2_q3_3():
    mean_v = np.zeros([22,26])
    count = np.zeros([22,1])
    for i in range(len(Xtrn)):
        mean_v[Ytrn[i]]+=Xtrn[i]
        count[Ytrn[i]]+=1
    mean_v = np.true_divide(mean_v,count)
    Z = linkage(mean_v,'ward')
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z, orientation='right', labels=abbvs)
    plt.xlabel('Distance')
    plt.ylabel('Languages')
    plt.show()

# iaml01cw2_q3_3()   # comment this out when you run the function

# Q3.4
def iaml01cw2_q3_4():
    big_cc=[]
    for i in np.unique(Ytrn):
        X = Xtrn[Ytrn==i]
        kmeans = skc.KMeans(n_clusters=3, random_state=1)
        kmeans.fit(X)
        cc = kmeans.cluster_centers_
        for c in cc:
            big_cc.append(c)
    big_cc = np.array(big_cc)
    lbls = ['cluster {} of language {}'.format(i,abbvs[j]) for j in range(22) for i in [1,2,3]]
    for method in ['ward','single','complete']:
        Z = linkage(big_cc,method)
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z, orientation='right', labels=lbls)
        plt.title('Dendogram for ' + method)
        plt.xlabel('Distance')
        plt.ylabel('Language clusters')
        plt.show()

# iaml01cw2_q3_4()   # comment this out when you run the function

# Q3.5
def iaml01cw2_q3_5():
    Xtrn_0 = np.array([Xtrn[j] for j in range(len(Xtrn)) if Ytrn[j]==0])
    Xtst_0 = np.array([Xtst[j] for j in range(len(Xtst)) if Ytst[j]==0])
    tabel = []
    for cov in ['diag', 'full']:
        row = []
        for k in [1,3,5,10,15]:
            gmm = mixture.GaussianMixture(n_components=k, covariance_type=cov, random_state=1)
            gmm.fit(Xtrn_0)
            train_avg = gmm.score(Xtrn_0)
            test_avg = gmm.score(Xtst_0)
            col = (train_avg, test_avg)
            row.append(col)
        tabel.append(row)
    tabel = np.array(tabel)
    print(tabel.shape)
    OX = [1,3,5,10,15]
    diag_cov_train = []
    diag_cov_test = []
    full_cov_train = []
    full_cov_test = []
    for i in range(len(tabel)):
        row = tabel[i]
        for j in range(len(row)):
            col=row[j]
            if i==0:
                diag_cov_train.append(col[0])
                diag_cov_test.append(col[1])
            else:
                full_cov_train.append(col[0])
                full_cov_test.append(col[1])
    
    # PLOT GRAPH
    plt.plot(OX,diag_cov_train, 'r', label='training set, diagonal covariance')
    plt.plot(OX,diag_cov_test, 'g', label='test set, diagonal covariance')
    plt.plot(OX,full_cov_train, 'b', label='training set, full covariance')
    plt.plot(OX,full_cov_test, 'y', label='test set, full covariance')
    plt.legend()
    plt.xlabel('Number of components')
    plt.ylabel('Average log-likelihood per sample')
    plt.show()

    # PLOT TABLE
    rows = ['Diagonal covariance', 'Full covariance']
    cols = ['{} components'.format(k) for k in [1,3,5,10,15]]
    plt.axis('off')
    table_vals = [["(train, test) = ({},{})".format(round(col[0],3),round(col[1],3)) for col in row] for row in tabel]
    my_table = plt.table(cellText = table_vals, colWidths = [0.12, 0.12,0.12,0.12,0.12], colLabels=cols, rowLabels=rows, loc = 'center')
    my_table.auto_set_font_size(False)
    my_table.set_fontsize(10)
    my_table.scale(1.5, 1.5)
    plt.show()

# iaml01cw2_q3_5()   # comment this out when you run the function

