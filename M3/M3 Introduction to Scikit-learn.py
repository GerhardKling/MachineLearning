"""
M3 Introduction to Scikit-learn
@author: GK
"""

#==================================================================================
#Scikit-learn - access to data
#==================================================================================
#https://scikit-learn.org/stable/

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

#Load data through API (application programming interface))
iris = datasets.load_iris()

#You will notice that the object iris is not just a data frame as in Pandas
#, e.g. information regarding the target variable is saved

y = iris.target
#We use the function target on the object iris, which basically assigns 
#y to the target variable in the dataset (i.e. the classification)

print('Class labels:', np.unique(y))
#The function np.unique is from Numpy and shows unique values 
#(the three categories)

#The following selects the last two columns of the dataset (explanatory variables)
X = iris.data[:, 2 : 4]

#Alternative approach
#X = iris.data[:, [2, 3]]

#==================================================================================
#Splitting the data into training and test sample
#==================================================================================

X_train, X_test, y_train, y_test = train_test_split(\
    X, y, test_size=0.3, random_state=1, stratify=y)

print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

#We randomly select 70% of the data as training and 30% as test data
#Stratification means that the proportions of categories is the same in 
#both sub-samples

#==================================================================================
#Standardization of explanatory variables (feature dimension)
#==================================================================================
#It is useful to standardize variables by removing their mean and 
#dividing by their standard deviation
#If the variable has a normal distribution, standardization ensures 
#that it is standard normal, i.e. N(0,1)
#Algorithms tend to work better with standardised data
#Have in mind that we make implicit assumptions here 
#(e.g. stationarity if dealing with time series data etc.)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#Note that fitting refers to the training data


#==================================================================================
#The perceptron in Scikit-learn
#==================================================================================
#Multiclass classification is supported using the one-versus-rest (OvR) 
#method; thus, we can train our perceptron to classify into three categories

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

#We specify 40 iterations (or epochs) and the learning rate is set at 0.1

#This checks the number of misclassifications we made
y_pred = ppn.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y_pred).sum())
#Note that %d acts as a placeholder for a number (in our case 3) 
#This number is passed via the % operator, which refers to the total 
#number of cases where predictions did not match the true labels

print(f"Misclassified samples: {(y_test != y_pred).sum()}")
#Equivalent approach using formatted string

#Note: the sum method works on arrays - but not on lists
(y_test != y_pred).sum()

np.array([True, True, False]).sum()

#Performance metrics
#Similarly, one can specify predictors via the score function

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))


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

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('Result.png', dpi=300)
plt.show()
