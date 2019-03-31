import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from pandas import Series

warnings.filterwarnings('ignore')

#read csv file
dataset = pd.read_csv('lte_cells_hourlyCSV.csv', parse_dates=True, dayfirst=True)

#make a dataframe
df = pd.DataFrame(dataset)

#change date fromat to datetime
df['DT'] = df['DT'].astype('datetime64[ns]')##df.sort_values('DT')


#make date index
df.set_index('DT', drop=True, append=False, inplace=True, verify_integrity=False)

#sort by date index
df = df.sort_index()

#groupby tower id
#g is a dataframe groupby object
## split

g = df.groupby('ENB_ID')

for towerid, tower_df in g:
    print(towerid)
    printn(tower_df)
    
#lets make different csv files with different ways of NA replacement for any 5 groups

group1014 = g.get_group(1014)


#first i make a csv file of the group itself with NA values
group1014.to_csv('group1014.csv')

#replace NA with 0 and make another csv file for the same group

print(group1014.isnull().values.any())

group1014.replace(np.nan, 0, inplace=True)

print(group1014.isnull().values.any())

#print(group1014)
group1014.to_csv('group1014NAtoZero.csv')

#drop rows with NA and make another csv file for the same group

group1014.dropna(inplace=True)

group1014.to_csv('group1014DroppedNA.csv')


#replace NA with means and make another csv file for the same group
print(group1014.mean())
group1014.fillna(group1014.mean(), inplace=True)
group1014.to_csv('group1014NAtoMean.csv')


#---------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------
print(group1014.dtypes)

