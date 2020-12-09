import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg as AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from mpl_toolkits import mplot3d

def get_data(filepath):

    dataset_train = pd.read_csv(filepath)

    features = dataset_train['last_price']
    original_data = features.values

    log_dataset = np.log(original_data)
    log_dif_dataset = np.diff(log_dataset, axis=0)

    return log_dif_dataset

def train_ar(dataset):

    model = AR(dataset, lags=5)
    results = model.fit()

    plt.plot(dataset[:20], color='blue')
    plt.plot(results.fittedvalues[:20], color='red')
    plt.show()

# https://www.statisticshowto.com/arma-model/
# p is the order of the autoregressive polynomial
# q is the order of the moving average polynomial
def train_arma(dataset):

    model = ARMA(dataset, order=(3,3))
    results = model.fit()
    plt.plot(dataset[200:])
    plt.plot(results.fittedvalues[200:], color='red')
    plt.show()

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def train_arima(dataset):

    model = model = ARIMA(dataset[:7000], order=(10,1,10))
    results = model.fit(disp=-1)
    results.plot_predict(6900,6950,dynamic=False)
    plt.show()

def train_var(dataset1,dataset2):

    index1 = 110000
    index2 = None

    df = pd.DataFrame({'x':dataset1[:index1],'y':dataset2[:index1]})

    model = VAR(df)
    results = model.fit(maxlags=100, ic='aic')

    # results.plot_forecast(5)
    # plt.show()

    # # show data
    # results.plot()
    # plt.show()

    # # plot autocorrelation functions
    # results.plot_acorr()
    # plt.show()

    lag_order = results.k_ar

    steps_ahead = 5

    index2 = index1 + steps_ahead

    forecast = results.forecast(df.values[-lag_order:], steps_ahead)

    labels = np.arange(start=0, stop=index2)
    plt.plot(labels[109990:], dataset1[109990:index2], color="blue")
    # plt.plot(labels[49900:index1], dataset2[49900:index1], color="orange")
    plt.plot(labels[index1:], forecast[:,0], '--', color="blue")
    plt.plot(labels[index1:], forecast[:,1], '--', color="orange")
    plt.show()

def train_varmax(dataset1,dataset2):

    index1 = 100000
    index2 = index1 + 10
    index3 = index1 - 10

    df = pd.DataFrame({'x':dataset1[:index1],'y':dataset2[:index1]})

    model = VARMAX(df)
    results = model.fit(maxlags=50, ic='aic')

    predictions = results.predict(index1,index2-1,False)

    print(predictions)

    labels = np.arange(start=0, stop=index2)
    plt.plot(labels[index3:], dataset1[index3:index2], color="blue")
    # plt.plot(labels[49900:index1], dataset2[49900:index1], color="orange")
    plt.plot(labels[index1:], predictions.values[:,0], '--', color="blue")
    # plt.plot(labels[index1:], predictions.values[:,1], '--', color="orange")
    plt.show()

def get_time_delay_embedding(dataset1,num_of_dims,interval_to_next_dim):

    embedded_points = []
    upper_bound = len(dataset1) - ((num_of_dims-1)*interval_to_next_dim)
    for x in range(upper_bound):
        current_sample = []

        for y in range(num_of_dims):
            index = x + (y*interval_to_next_dim)
            current_sample.append(dataset1[index])



        embedded_points.append(current_sample.copy())

    print(len(embedded_points))

    return np.array(embedded_points)





if __name__ == "__main__":

    filepath1 = 'intraday_minute_data2.csv'
    log_dif_dataset1 = get_data(filepath1)


    # filepath2 = 'intraday_minute_data2.csv'
    # log_dif_dataset2 = get_data(filepath2)

    embedded_points = get_time_delay_embedding(log_dif_dataset1,3,1)



