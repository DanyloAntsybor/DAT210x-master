import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#

saples = []

#
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#

filepath = 'D:\\work\\Courses\\DAT210x-master\\Module4\\Datasets\\ALOI\\32'
listfiles = [f for f in listdir(filepath) if isfile(join(filepath, f))]

# Load the image up
dset = []
colors = []
for fname in listfiles:
  img = misc.imread(join(filepath, fname))
  dset.append(img.reshape(-1))
  colors.append('r')


#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#

filepath1 = 'D:\\work\\Courses\\DAT210x-master\\Module4\\Datasets\\ALOI\\32i'
listfiles1 = [f for f in listdir(filepath1) if isfile(join(filepath1, f))]

# Load the image up

for fname in listfiles1:
  img = misc.imread(join(filepath1, fname))
  dset.append(img.reshape(-1))
  colors.append('b')


#
# TODO: Convert the list to a dataframe
#
saples = pd.DataFrame( dset )



#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
from sklearn import manifold
iso = manifold.Isomap(n_neighbors=6, n_components=3)
T_iso = iso.fit_transform(saples)


#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(T_iso[:,0],T_iso[:,1], c=colors, marker='.',alpha=0.7)





#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 
fig1 = plt.figure()
ax2 = fig1.add_subplot(111, projection='3d')

ax2.scatter(T_iso[:,0], T_iso[:,1], T_iso[:,2], c=colors, marker='.')

ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')
ax2.set_zlabel('Z Label')



plt.show()

