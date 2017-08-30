import pandas as pd

# TODO: Load up the 'tutorial.csv' dataset
#D:\work\Courses\DAT210x-master\Module2\Datasets\tutorial.csv
filepath = 'D:\\work\\Courses\\DAT210x-master\\Module2\\Datasets\\tutorial.csv'
test_frame = pd.read_csv(filepath)



# TODO: Print the results of the .describe() method
#
print(test_frame.describe())



# TODO: Figure out which indexing method you need to
# use in order to index your dataframe with: [2:4,'col3']
# And print the results
#
print(test_frame.loc[2:4,['col3']])
