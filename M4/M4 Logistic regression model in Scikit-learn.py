"""
M4 Logistic regression model in Scikit-learn
@author: GK
"""

#==================================================================================
#Logistic regression model in Scikit-learn
#==================================================================================
#https://scikit-learn.org/stable/

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

#Step 1: Load data
iris = datasets.load_iris()

#Step 2: Define output 
y = iris.target

#Step 3: Define data matrix
X = iris.data[:, 2 : ]

#Step 4: Split the sample
X_train, X_test, y_train, y_test = train_test_split(\
    X, y, test_size=0.3, random_state=1, stratify=y)
    
#Step 5: Standardise explanatory variables (inputs)    
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)    

lr = LogisticRegression()
lr.fit(X_train_std, y_train)

#==================================================================================
#Visualization
#==================================================================================
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
    # highlight test samples
    if test_idx:
        # plot all samples
        X_test = X[test_idx, :]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')   

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('Logistic.png', dpi=300)
plt.show()

#The model returns predicted probabilities for each observation to be classified 
#as one of the three classes
#The following code shows probabilities for the first three observations
lr.predict_proba(X_test_std[:3, :])
#By construction, the predicted probabilities for each observation sum up to one
#, as there are only three outcomes
