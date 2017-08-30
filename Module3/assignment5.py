#
# This code is intentionally missing!
# Read the directions on the course lab page!
#

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pandas.tools.plotting import andrews_curves

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
filepath = 'D:\\work\\Courses\\DAT210x-master\\Module3\\Datasets\\wheat.data'
wheat = pd.read_csv(filepath, index_col = 0)


#
# TODO: Drop the 'id' feature, if you included it as a feature
# (Hint: You shouldn't have)
# Also get rid of the 'area' and 'perimeter' features
# 
#wheat.drop(['area','perimeter'], axis = 1, inplace = True)



#
# TODO: Plot a parallel coordinates chart grouped by
# the 'wheat_type' feature. Be sure to set the optional
# display parameter alpha to 0.4
# 

plt.figure()
andrews_curves(wheat, 'wheat_type', alpha = 0.4)

plt.show()


