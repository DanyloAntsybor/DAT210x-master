import pandas as pd


# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading

html_str = 'http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2'
nhl_frame = pd.read_html(html_str, header = 1)
nhl_frame = nhl_frame[0]



# TODO: Rename the columns so that they are similar to the
# column definitions provided to you on the website.
# Be careful and don't accidentially use any names twice.

nhl_frame.columns = ['RK', 'PLAYER', 'TEAM', 'GP', 'G', 'A', 'PTS', 'PLUSMINUS',
        'PIM', 'PTS_G','SOG', 'PCT', 'GWG', 'PP_G', 'PP_A', 'SH_G', 'SH_A']


# TODO: Get rid of any row that has at least 4 NANs in it,
# e.g. that do not contain player points statistics

nhl_frame = nhl_frame.dropna(axis=0, thresh= len(nhl_frame.columns) - 4)

# TODO: Get rid of the 'RK' column
nhl_frame = nhl_frame.drop(axis=1, labels=['RK'])

# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?

nhl_frame = nhl_frame[(nhl_frame['PLAYER']!='PLAYER')&(nhl_frame['TEAM']!='TEAM')]

# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index

nhl_frame.reset_index(inplace = True, drop = True)

# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric

nhl_frame['GP'] = pd.to_numeric(nhl_frame['GP'], errors='coerce')
nhl_frame['G'] = pd.to_numeric(nhl_frame['G'], errors='coerce')
nhl_frame['A'] = pd.to_numeric(nhl_frame['A'], errors='coerce')
nhl_frame['PTS'] = pd.to_numeric(nhl_frame['PTS'], errors='coerce')
nhl_frame['PLUSMINUS'] = pd.to_numeric(nhl_frame['PLUSMINUS'], errors='coerce')
nhl_frame['PIM'] = pd.to_numeric(nhl_frame['PIM'], errors='coerce')
nhl_frame['PTS_G'] = pd.to_numeric(nhl_frame['PTS_G'], errors='coerce')
nhl_frame['SOG'] = pd.to_numeric(nhl_frame['SOG'], errors='coerce')
nhl_frame['PCT'] = pd.to_numeric(nhl_frame['PCT'], errors='coerce')
nhl_frame['GWG'] = pd.to_numeric(nhl_frame['GWG'], errors='coerce')
nhl_frame['PP_G'] = pd.to_numeric(nhl_frame['PP_G'], errors='coerce')
nhl_frame['PP_A'] = pd.to_numeric(nhl_frame['PP_A'], errors='coerce')
nhl_frame['SH_G'] = pd.to_numeric(nhl_frame['SH_G'], errors='coerce')
nhl_frame['SH_A'] = pd.to_numeric(nhl_frame['SH_A'], errors='coerce')

# TODO: Your dataframe is now ready! Use the appropriate 
# commands to answer the questions on the course lab page.
#
len(nhl_frame.PCT.unique())
nhl_frame.GP[15] + nhl_frame.GP[16]
