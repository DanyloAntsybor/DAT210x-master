import pandas as pd
import matplotlib.pyplot as plt


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
filepath = 'D:\\work\\Courses\\DAT210x-master\\Module3\\Datasets\\wheat.data'
wheat = pd.read_csv(filepath, index_col = 0)

#
# TODO: Drop the 'id' feature, if you included it as a feature
# (Hint: You shouldn't have)
# 
# .. your code here ..


#
# TODO: Compute the correlation matrix of your dataframe
# 
# .. your code here ..


#
# TODO: Graph the correlation matrix using imshow or matshow
# 

plt.imshow(wheat.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(wheat.columns))]
plt.xticks(tick_marks, wheat.columns, rotation= 45)
plt.yticks(tick_marks, wheat.columns)

plt.show()

plt.show()


