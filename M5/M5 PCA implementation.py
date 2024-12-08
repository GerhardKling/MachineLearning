"""
M5 PCA implementation
@author: GK
"""

#==================================================================================
#PCA in Scikit-learn
#==================================================================================
#https://scikit-learn.org/stable/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

#==================================================================================
#Load data
#==================================================================================
#We load some data on wine, which is available online, using Pandas

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

#Name variables
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

#Next we define the data matrix and target variable (label) 
#NOTE: iloc refers to integer-location based indexing for selection by position
X = df_wine.iloc[:, 1:].values
y = df_wine.iloc[:, 0].values

#Alternative approach using series
# y = df_wine['Class label']


#==================================================================================
#Split sample
#==================================================================================
#Then we split the sample into 70% training and 30% test data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size  = 0.3, 
    stratify=y,
    random_state=0)

#==================================================================================
#Standardisation
#==================================================================================
#Standardise data matrix X
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#==================================================================================
#Eigendecomposition of the covariance matrix
#==================================================================================
#We call the Numpy library and calculate the covariance matrix for the 
#standardised variables in the data matrix X, which belong to the training 
#dataset. Using the linear algebra package, we eigenvalues and their 
#associated eigenvectors

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

#It is useful to note that the sum of all eigenvalues is equal to the total 
#explained variance of the original data
#Each principal component’s contribution to explaining the total variance 
#refers to its eigenvalue relative to the total explained variance
#The Numpy np.cumsum() refers to a cumulated sum starting with one element 
#and adding another 
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

#We plot the sorted eigenvalues and their explanatory power in terms of 
#explained variance
#Hence, restricting us to a few eigenvalues and their associated eigenvectors 
#is sufficient to explain a large proportion of the total variance
#Note that plt.step() refers to a step function
plt.figure(1)
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('Fig1.png', dpi=300)
plt.show()


#==================================================================================
#Defining the transformation matrix
#==================================================================================
#We select only the two most important principal components
#The associated eigenvectors define the transformation matrix W
#List of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

#Sort the (eigenvalue, eigenvector) tuples from high to low
#Note: lambda is an anonymous callable function; k is parameter of function

eigen_pairs.sort(reverse=True)
# eigen_pairs.sort(key=lambda k: k[0], reverse=True)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)

eigen_pairs.sort(reverse=True)

#Note: np.newaxis increases dimension by one; transpose


#==================================================================================
#Plotting our results
#==================================================================================
#We simplified a higher dimensional problem to a low dimensional problem
#Looking at the labels, we should be able to use our machine learning tools 
#(see previous lectures) to separate the subgroups
#List of (eigenvalue, eigenvector) tuples

X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

plt.figure(2)
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('Fig2.png', dpi=300)
plt.show()
