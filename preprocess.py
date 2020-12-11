import scipy.io
import os
import examine_data
from scipy.stats import boxcox
import numpy as np
import train_ssm

def make_stationary(dataset):
    '''Returns transformed dataset that is stationary.'''

    # remove negatives and make min val 1
    dataset += 1 + (-1*np.min(dataset))

    # boxcox transformation to dynamically remove trends
    dataset, lam = boxcox(dataset)

    # found two worked best for theta band
    # 2nd order differencing to enforce stationarity on adf test w/o over differencing
    dataset = np.diff(dataset, axis=0)
    dataset = np.diff(dataset, axis=0)

    return dataset, lam, min(dataset)

DATA_FOLDER_PATH = '/home/matt/eeg_data/power/LNR/evoked'
FCz_data = scipy.io.loadmat(DATA_FOLDER_PATH + "/" + "POWevoked_leftnorew_FC1.mat")


# eeg_data = []
# counter = 0
# for f in os.listdir(DATA_FOLDER_PATH):

#     if 'FCz' in f:
#         continue
#     else:
#         eeg_data.append(scipy.io.loadmat(DATA_FOLDER_PATH + "/" + f))
        
#     counter += 1
#     if counter == 3:
#         break

feedback_onset_FCz_theta =  FCz_data['POW_evoked'][:,3:7,626:888]
for i in range(feedback_onset_FCz_theta.shape[0]):
    for j in range(feedback_onset_FCz_theta.shape[1]):
        current_FCz_theta_compare = feedback_onset_FCz_theta[i,j]
        current_FCz_theta_compare,_,_ = make_stationary(current_FCz_theta_compare)
        # np.savetxt("FC1_theta_data.csv", current_FCz_theta_compare, delimiter=",")
        with open("FC1_theta_data.csv", 'ab') as abc:
            np.savetxt(abc, current_FCz_theta_compare, delimiter=",")
# train_ssm.train_arma(current_FCz_theta_compare)
