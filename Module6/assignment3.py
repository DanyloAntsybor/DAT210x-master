#
# This code is intentionally missing!
# Read the directions on the course lab page!
#
import pandas as pd
import numpy as np
from sklearn import preprocessing

#Load up the /Module6/Datasets/parkinsons.data data set into a variable X,
#being sure to drop the name column.
filepath = 'D:\\work\\Courses\\DAT210x-master\\Module6\\Datasets\\parkinsons.data'
X = pd.read_csv(filepath)
X.drop('name', axis = 1, inplace = True)
print("Is there any Nan:")
print(X.isnull().any().any())

#Slice out the status column into a variable y and delete it from X.
y = X['status'].copy()
X.drop('status', axis = 1, inplace = True)

#Perform a train/test split. 30% test group size, with a random_state equal to 7.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


#Create a SVC classifier.
#Don't specify any parameters, just leave everything as default.
#Fit it against your training data and then score your testing data.

from sklearn.svm import SVC
#kernel , C, gamma
svc = SVC()
svc.fit(X_train, y_train)
score = svc.score(X_test, y_test)

print("Score SVC model with initial parameters:")
print(score)

#That accuracy was just too low to be useful. We need to get it up.
#Program a naive, best-parameter search by creating nested for-loops.
#The outer for-loop should iterate C from 0.05 to 2, using 0.05 increments.
#The inner for-loop should increment gamma from 0.001 to 0.1, using 0.001 increments.
#As you know, Python ranges won't allow for float intervals,
#so you'll have to do some research on NumPy ARanges, if you don't already know how to use them.

#Since the goal is to find the parameters that result in the model
#having the best accuracy score, you'll need a best_score = 0 variable
#that you initialize outside of the for-loops.
#Inside the inner for-loop, create an SVC model
#and pass in the C and gamma parameters its class constructor.
#Train and score the model appropriately.
#If the current best_score is less than the model's score,
#update the best_score being sure to print it out,
#along with the C and gamma values that resulted in it.

prep = preprocessing.StandardScaler().fit(X_train) #0.932203389831
#prep = preprocessing.MinMaxScaler().fit(X_train) #0.881355932203
#prep = preprocessing.MaxAbsScaler().fit(X_train) #0.881355932203
#prep = preprocessing.Normalizer().fit(X_train) #0.796610169492
#prep = preprocessing.KernelCenterer().fit(X_train) #0.915254237288
#prep = preprocessing.RobustScaler().fit(X_train) #0.915254237288

X_train = prep.transform(X_train)
X_test = prep.transform(X_test)

best_score = 0

#from sklearn.decomposition import PCA #0.932203389831 max score
X_train_prep = X_train
X_test_prep = X_test
#n_component values between 4 and 14
#for n_comp in range(4,15):
#    pca = PCA(n_components = n_comp, svd_solver='auto', random_state=7)
#    pca.fit(X_train_prep)
#    X_train = pca.transform(X_train_prep)
#    X_test = pca.transform(X_test_prep)
#### IsoMap ####
from sklearn import manifold #0.949152542373
iterr = np.array([[[n,c] for c in range(4,7)] for n in range(2,6)])
iterr = iterr.reshape(12,2)
for i in iterr:
    iso = manifold.Isomap(n_neighbors=i[0], n_components=i[1])
    iso.fit(X_train_prep)
    X_train = iso.transform(X_train_prep)
    X_test = iso.transform(X_test_prep)
    for C in np.arange(0.05, 2.05, 0.05):
        for gamma in np.arange(0.001, 0.101, 0.001):
            svc = SVC(kernel = 'rbf', C = C, gamma = gamma)
            svc.fit(X_train, y_train)
            score = svc.score(X_test, y_test)
            if score > best_score:
                best_score = score

print(best_score)

