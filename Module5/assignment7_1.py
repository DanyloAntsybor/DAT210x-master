# If you'd like to try this lab with PCA instead of Isomap,
# as the dimensionality reduction technique:
from sklearn import preprocessing
import pandas as pd
import numpy as np

Test_PCA = False


def plotDecisionBoundary(model, X, y):
  print ("Plotting...")
  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.style.use('ggplot') # Look Pretty

  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.1
  resolution = 0.1

  #(2 for benign, 4 for malignant)
  colors = {2:'royalblue',4:'lightsalmon'} 

  
  # Calculate the boundaris
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Create a 2D Grid Matrix. The values stored in the matrix
  # are the predictions of the class at at said location
  import numpy as np
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
  plt.axis('tight')

  # Plot your testing points as well...
  for label in np.unique(y):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], alpha=0.8)

  p = model.get_params()
  plt.title('K = ' + str(p['n_neighbors']))
  plt.show()


# 
# TODO: Load in the dataset, identify nans, and set proper headers.
# Be sure to verify the rows line up by looking at the file in a text editor.
#
filepath = 'D:\\work\\Courses\\DAT210x-master\\Module5\\Datasets\\breast-cancer-wisconsin.data'
col_names = ['sample_number', 'thickness', 'size', 'shape',
'adhesion', 'epithelial', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status']
df = pd.read_csv(filepath, index_col = 0, header = None, names = col_names, na_values=["?"])


# 
# TODO: Copy out the status column into a slice, then drop it from the main
# dataframe. Always verify you properly executed the drop by double checking
# (printing out the resulting operating)! Many people forget to set the right
# axis here.
#
# If you goofed up on loading the dataset and notice you have a `sample` column,
# this would be a good place to drop that too if you haven't already.
#
status = df['status'].copy()
df.drop(labels=['status'], inplace=True, axis=1)



#
# TODO: With the labels safely extracted from the dataset, replace any nan values
# with the mean feature / column value
#
df = df.fillna(df.mean())


#
# TODO: Do train_test_split. Use the same variable names as on the EdX platform in
# the reading material, but set the random_state=7 for reproduceability, and keep
# the test_size at 0.5 (50%).
#
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, status, test_size=0.5, random_state=7)


#
# TODO: Experiment with the basic SKLearn preprocessing scalers. We know that
# the features consist of different units mixed in together, so it might be
# reasonable to assume feature scaling is necessary. Print out a description
# of the dataset, post transformation. Recall: when you do pre-processing,
# which portion of the dataset is your model trained upon? Also which portion(s)
# of your dataset actually get transformed?
#
# TODO: Un-comment just ***ONE*** of lines at a time and see how alters your results

#prep = preprocessing.StandardScaler().fit(X_train)
#prep = preprocessing.MinMaxScaler().fit(X_train)
#prep = preprocessing.MaxAbsScaler().fit(X_train)
prep = preprocessing.Normalizer().fit(X_train)
#prep = preprocessing.RobustScaler().fit(X_train)
#prep = X_train # No Change

X_train = prep.transform(X_train)
X_test = prep.transform(X_test)


# PCA and Isomap are your new best friends
model = None

from sklearn.decomposition import PCA
from sklearn import manifold

prediction_list = []
# 4 for PCA and 5-10 for IsoMap
for test in range(4,11):
  if test == 4:
    print ("PCA")
    model = PCA(n_components = 2, svd_solver='randomized', random_state=7)
  else:
    print ("Isomap " + str(test))
    model = manifold.Isomap(n_neighbors=test, n_components=2)

  model.fit(X_train)
  X_train = model.transform(X_train)
  X_test = model.transform(X_test) 

  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.metrics import accuracy_score

  
  current = np.array([])
  for i in range(1,16):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    prediction_list.append(accuracy_score(y_test, predictions))
    current = np.append(current,accuracy_score(y_test, predictions))
  print(current.max())

prediction_list = np.array(prediction_list)
print("Final: ")
print(prediction_list.max())

#plotDecisionBoundary(knn, X_test, y_test)
