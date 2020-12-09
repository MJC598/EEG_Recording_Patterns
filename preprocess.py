import scipy.io
import os
import examine_data
from scipy.stats import boxcox
import numpy as np
import train_ssm

def make_stationary(dataset):
    '''Returns transformed dataset that is stationary.'''

    # remove negatives and make min val 1
    dataset += 1 + (-1*min(dataset))

    # boxcox transformation to dynamically remove trends
    dataset, lam = boxcox(dataset)

    # found two worked best for theta band
    # 2nd order differencing to enforce stationarity on adf test w/o over differencing
    dataset = np.diff(dataset, axis=0)
    dataset = np.diff(dataset, axis=0)

    return dataset, lam, min(dataset)

DATA_FOLDER_PATH = 'D:\power_out2\LNR\evoked'
FCz_data = scipy.io.loadmat(DATA_FOLDER_PATH + "/" + "POWevoked_leftnorew_FCz.mat")


eeg_data = []
counter = 0
for f in os.listdir(DATA_FOLDER_PATH):

    if 'FCz' in f:
        continue
    else:
        eeg_data.append(scipy.io.loadmat(DATA_FOLDER_PATH + "/" + f))
        
    counter += 1
    if counter == 3:
        break

feedback_onset_FCz_theta =  FCz_data['POW_evoked'][:,3:7,626:888]
current_FCz_theta_compare = feedback_onset_FCz_theta[0,0]
current_FCz_theta_compare,_,_ = make_stationary(current_FCz_theta_compare)
train_ssm.train_arma(current_FCz_theta_compare)
