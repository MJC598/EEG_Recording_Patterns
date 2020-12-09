import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import coint
import math
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# examine auto-correlation 
'''
We can calculate the correlation for time series observations with observations with previous time steps, called lags. 
Because the correlation of the time series observations is calculated with values of the same series at previous times, this is called a serial correlation, or an autocorrelation.

We use ACF and PACF to choose a correct order for AR(p) and MA(q) components/features of an ARIMA model. For AR order p, look at PACF plot and choose a lag value which has a significant correlation factor before correlations get insignificant. For MA order q look at ACF plot and do the same. Don’t forget you should only get these values from the ACF and PACF plots of stationary time series
'''
def get_autocorrelations(log_dif_dataset):

    plot_acf(log_dif_dataset, lags=30, fft=True)
    plt.show()

    # plot_pacf(log_dif_dataset, lags=30)
    # plt.show()

# Dickey–Fuller test:
'''
# Tests for a unit root in the time series sample.
# The augmented version is for larger and more complicated set of time series models
# The augmented Dickey–Fuller (ADF) statistic, used in the test, is a negative number. The more negative it is, the stronger the rejection of the hypothesis that there is a unit root at some level of confidence
'''
def get_adf_test(log_dif_dataset):
    result = adfuller(log_dif_dataset)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

# rolling statistics plot
def get_rolling_stats_plot(log_dif_dataset):

    df = pd.DataFrame(data=log_dif_dataset)
    rolling_mean = df.rolling(window=20).mean()
    rolling_std = df.rolling(window=20).std()
    original = plt.plot(df, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('log of original data with difference between daily close')
    plt.show()


# the null hypothesis is that there is no cointegration
# the more negative, then the stronger the null hypothesis is rejected
def get_cointegration_test(dataset1,dataset2):

    # log_dataset = np.log(dataset1)
    temp_dataset1 = np.diff(dataset1, axis=0)
    
    # log_dataset = np.log(dataset2)
    temp_dataset2 = np.diff(dataset2, axis=0)

    result = coint(temp_dataset1,temp_dataset2)
    # print('Cointegration Statistic: {}'.format(result[0]))
    # print('p-value: {}'.format(result[1]))
    # print('Critical Values:')
    # print("1%: {}".format(result[2][0]))
    # print("5%: {}".format(result[2][1]))
    # print("10%: {}".format(result[2][2]))

    return result[0], result[1], result[2][0]
    