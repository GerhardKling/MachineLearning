"""
Classification problem

"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np


#Load data
iris = datasets.load_iris()

#Define target variable
y = iris.target

#Classification
print('Class labels:', np.unique(y))

#Data matrix
X = iris.data[:, [2, 3]]

#Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

#Standardisation
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Logistic regression
lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)

#Combine testing and training data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.concatenate((y_train, y_test)).flatten()

plot_decision_regions(X_combined_std, y_combined,
                      clf=lr, X_highlight=X_combined_std[105:150])
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('Logistic.png', dpi=300)
plt.show()

#Show probabilities
print(lr.predict_proba(X_test_std[:3, :]))



