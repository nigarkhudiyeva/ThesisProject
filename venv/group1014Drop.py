import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
from pandas import DataFrame
#from sklearn import linear_model
#from sklearn.metrics import mean_squared_error
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import autocorrelation_plot



warnings.filterwarnings('ignore')

dataset = pd.read_csv('group1014DroppedNA.csv', parse_dates=True, dayfirst=True)
df = pd.DataFrame(dataset)

ts = pd.Series(df['INTERF_PUCCH'].values, index=df['DT'])
#print(ts.head(5))

#ts.plot()
#plt.show()


train_size = int(len(ts) * 0.80)
train, test = ts[0:train_size], ts[train_size:len(ts)]
print('Observations: %d' % (len(ts)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))

from statsmodels.tsa.stattools import adfuller
ts_log = np.log(train)

def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(window=24).mean()  # 24 hours on each day
    rolstd = timeseries.rolling(window=24).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print
    ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

#test_stationarity(train)



#print(train.head())
#train.head(40).plot()
#plt.show()
#autocorrelation_plot(train.head(20))
#plt.show()



model = ARIMA(train, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


# ts_log = np.log(train)
# plt.plot(ts_log)
# #plt.show()
#
# moving_avg = train.rolling(ts_log,12).mean()
# plt.plot(ts_log)
# plt.plot(moving_avg, color='red')
#plt.plot(train)
#plt.plot([None for i in train] + [x for x in test])
#plt.show()

# ts_log = np.log(train)
# #print(ts_log)
#
# ts_log_diff = ts_log - ts_log.shift()
# #
# model = ARIMA(ts_log, order=(2, 1, 0))
# results_AR = model.fit(disp=-1)
# plt.plot(ts_log_diff)
# plt.plot(results_AR.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
#print(train.isnull().values.any())
#df['DT'] = df['DT'].astype('datetime64[ns]')##df.sort_values('DT')
#print(train['2018-12-23 00:00:00'])
