# Import libraries
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = 15, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA


# Import NHS inpatient and outpatient data, skipping irrelevant columns
df = pd.read_csv('./data/QAR-COMM-Timeseries-1617-Q3-50207.csv', skiprows=13, usecols=[x for x in range(1,17) if x not in [3,4,9,10]])


# Convert Year and Period to time series data and create new column "-quarter" to use as index later
def getMonthYear(row):
    month=row['Period']
    year=int(row['Year'].split('-')[0])
    if month in ['MARCH']:
        year= year+1
    #Following the conversion, the _quarter year specifies the calendar year in which the financial year ends
    return pd.to_datetime("01-{}-{}".format(month[:3],year),format='%d-%b-%Y')

df['Financial_YQ']=pd.PeriodIndex(df.apply(getMonthYear,axis=1), freq='Q-MAR')
df[['Year','Period','Financial_YQ']].head()
#Note the syntax - the _quarter year specifies the financial end year

df = df.set_index('Financial_YQ')


# Review if the start and end dates of the corresponding periods are correct
print('Period Start Dates: ', df.index.asfreq('D', 's'))
print('Period Start Dates: ', df.index.asfreq('D', 'e'))


df1 = df[['Decisions to Admit', 'Admissions', 'GP Referrals Made', 'Other Referrals Made', 'First Attendances Seen', 'Subsequent Attendances Seen']]


# Visualise df1
df1.plot()
plt.title('NHS Outpatient and Inpatient Data Selection')
plt.ylabel('Patients in Million per Quarter')
plt.savefig('./plots/selected_timeseries.png')
plt.show(block=True)
plt.close()

# Define function to test stationarity based on rolling statistics and Dickey-Fuller test, set window = 4 for quarters
def test_stationarity(timeseries, plotname):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window=4, center=False).mean()
    rolstd = timeseries.rolling(window=4, center=False).std()

    #Plot rolling statistics:
    timeseries.plot(color='blue', label='Original')
    rolmean.plot(color='red', label='Rolling Mean')
    rolstd.plot(color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation for Dataset: %s' % plotname)
    plt.savefig('./plots/roll_mean_std_deviation_%s.png' % plotname)
    plt.show(block=True)
    plt.close()

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, maxlag=4)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

# Test stationarity, obviously this original NHS set is not stationary because of positive trend and light seasonality
# !!! I chose 'First Attendances Seen' for the further investigations because it might be interesting for doctors
# to use and compare the trend and forecast with their own surgery / practice data for better capacity planning

# Convert Index from period into timestamp before, because period is not supported by seasonal_decompose()
# I use only the last 16 quarters, because they seem to have more consistent and relevant data
selection = 'First Attendances Seen'
fa = df1[selection].tail(16)
fa.index = fa.index.to_timestamp(how='end')

# Next step: Make data as stationary as possible for further investigations
# Goal: isolate trend and seasonality in the series to get a stationary series
# Then do forecast on stationary time series and finally apply trend and seasonality back on forecast

# 1. Try Log Transformation to penalise higher values more than lower values:
fa_log = np.log(fa)

# Visualise results
fa_log.plot()
plt.title('Log Transformation for %s' % selection)
plt.savefig('./plots/log_transformation_%s.png' % selection)
plt.show(block=True)
plt.close()

# 2. Make it smoother with rolling average, taking average of last 4 quarters
#moving_avg = pd.rolling_mean(fa_log,4)
moving_avg = fa_log.rolling(window=4, center=False).mean()
fa_log.plot(label='Log Transformation')
moving_avg.plot(label='Moving Avg', color='red')
plt.legend(loc='best')
plt.title('Log Transformation and Moving Avg for %s' % selection)
plt.savefig('./plots/moving_avg_log_transformation_%s.png' % selection)
plt.show(block=True)
plt.close()

# 3. Sustract trend from data
fa_log_moving_avg_diff = fa_log - moving_avg

# 4. Drop NaN values (first 3 in our case) and check if data if stationary (very good in this)
fa_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(fa_log_moving_avg_diff, 'Moving Avg. Diff. of fa_log')

# 5. Decompose
# Might throw VisibleDeprecationWarning with statsmodels versions 0.6.* and older, but checked it and it disappears with 0.8
decomposition = seasonal_decompose(fa_log)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
decomposition.plot()
plt.savefig('./plots/decomposition_%s.png' % selection)
plt.show(block=True)
plt.close()

fa_log_decompose = residual
fa_log_decompose.dropna(inplace=True)
test_stationarity(fa_log_decompose, 'Decompose of fa_log')

# 6. ARIMA (AutoRegressive Integrated Moving Average) to handle autocorrelation, non-stationarity, and seasonality

# 6.1 ARIMA Do ACF and PACF plots to identify parameters for ARIMA model

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(fa_log_decompose, nlags=8)
lag_pacf = pacf(fa_log_decompose, nlags=8, method='ols')
#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(fa_log_decompose)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(fa_log_decompose)),linestyle='--',color='gray')
plt.title('Autocorrelation Function for: %s' % selection)
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(fa_log_decompose)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(fa_log_decompose)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.savefig('./plots/acf_pacf_%s.png' % selection)
plt.show(block=True)
plt.close()

# In this plot, the two outer dotted lines the confidence intervals.
# These can be used to approximately determine the ‘p’ and ‘q’ values as:
# p – The lag value where the PACF chart crosses the upper confidence interval for the first time.
# --> In this case p=1
# q – The lag value where the ACF chart crosses the upper confidence interval for the first time.
# --> In this case q=1
# ------
# d - The order of differencing should be set: d=0, if original series is stationary (no), d=1 if it has a constant
# average trend (yes), or d=2 if original series has a time-varying trend (not so much)
# --> We set d=1

# 6.2 Perform ARIMA(p,d,q)
model = ARIMA(fa_log, order=(1, 1, 1))
results_arima = model.fit(disp=-1)
fa_log_moving_avg_diff.plot(label='Log Moving Avg Diff')
results_arima.fittedvalues.plot(label='ARIMA Results', color='red')
plt.legend(loc='best')
plt.title('ARIMA Results')
plt.savefig('./plots/ARIMA_raw_%s.png' % selection)
plt.show(block=True)
plt.close()

# Finally apply trend and seasonality back on results_ARIMA
predictions_arima_diff = pd.Series(results_arima.fittedvalues, copy=True)
predictions_arima_diff_cumsum = predictions_arima_diff.cumsum()
predictions_arima_log = pd.Series(fa_log.ix[0], index=fa_log.index)
predictions_arima_log = predictions_arima_log.add(predictions_arima_diff_cumsum, fill_value=0)
predictions_arima = np.exp(predictions_arima_log)

# Check if the prediction algorithm is plausible and not overfitting the data set - looks good here
fa.plot(label='Actual Data')
predictions_arima.plot(label='Predictions ARIMA')
plt.legend(loc='best')
plt.ylabel('Patients per Quarter')
ax = plt.gca()
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.title('ARIMA Plausibility Check for: %s' % selection)
plt.savefig('./plots/ARIMA_plausibility_%s.png' % selection)
plt.show(block=True)
plt.close()

# Predict future values
# Extend Data Frame (I've hardcoded the dates here - It could be done more elegantly in the future)
start = dt.datetime.strptime("2017-03-31", "%Y-%m-%d")
date_list = pd.to_datetime(['2017-03-31', '2017-06-30', '2017-09-30', '2017-12-31', '2018-03-31'])
future = pd.DataFrame(index=date_list, columns=['Forecast'])
fa = pd.concat([fa, future])

# Calculate predicted values and put them back into original scale
future_predictions = results_arima.predict(start = 16, end = 21, dynamic= True)
print(future_predictions)
future_predictions_cunsum = future_predictions.cumsum()
future_predictions_log = pd.Series(fa_log.ix[15], index=future_predictions.index)
future_predictions_log = future_predictions_log.add(future_predictions_cunsum, fill_value=0)
future_predictions_fin = np.exp(future_predictions_log)
fa['Forecast'] = future_predictions_fin
fa.columns = ['Actual Data', 'Forecast']

# Finally print our new dataframe incl. predictions and plot forecast
fa.plot()
plt.legend(loc='best')
plt.ylabel('Patients per Quarter')
ax = plt.gca()
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.title('ARIMA Forecast for: %s' % selection)
plt.savefig('./plots/ARIMA_forecast_%s.png' % selection)
plt.show(block=True)
plt.close()

print(fa)