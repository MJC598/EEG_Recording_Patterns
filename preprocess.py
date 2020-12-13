import scipy.io
import os
import examine_data
from scipy.stats import boxcox
from statsmodels.graphics.correlation import plot_corr
import numpy as np
import matplotlib.pyplot as plt
import train_ssm
import csv
import random

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

# can also make a windowed version of this to look at more local dynamics
def get_TLCC(dataset1,dataset2,lag=0):
    '''Returns the time lag cross-correlation between two time-series.'''

    return np.corrcoef(np.roll(dataset1,lag),dataset2)

def get_corr_data():

    DATA_FOLDER_PATH = 'D:\power_out2\LNR\evoked'
    FCz_data = scipy.io.loadmat(DATA_FOLDER_PATH + "/" + "POWevoked_leftnorew_FCz.mat")

    eeg_data = []
    eeg_filenames = []
    counter = 0
    for f in os.listdir(DATA_FOLDER_PATH):

        if 'FCz' in f:
            continue
        else:
            eeg_filenames.append(f)
            eeg_data.append(scipy.io.loadmat(DATA_FOLDER_PATH + "/" + f))
            
        counter += 1
        print(counter)

    feedback_onset_FCz_theta =  FCz_data['POW_evoked'][:,3:7,626:888]
    # current_FCz_theta_compare = feedback_onset_FCz_theta[0,0]
    # current_FCz_theta_compare,_,_ = make_stationary(current_FCz_theta_compare)
    # train_ssm.train_arma(current_FCz_theta_compare)



    corr_list = []
    freq_list = []
    theta_list = []
    subject_list = []
    node_region_list = []
    for node_index,node_region in enumerate(eeg_data):
        print(str(node_index))
        for sub_index, subject in enumerate(node_region['POW_evoked']):
            # print(str(sub_index))
            for freq_index, freq in enumerate(subject):
                
                temp_freq = freq[626:888]
                temp_freq,_,_ = make_stationary(temp_freq)
                for theta_freq in range(4):
                    current_FCz_theta_compare =  feedback_onset_FCz_theta[sub_index,theta_freq]
                    current_FCz_theta_compare,_,_ = make_stationary(current_FCz_theta_compare)
                    corr = np.corrcoef(temp_freq,current_FCz_theta_compare)

                    rs = [str(get_TLCC(temp_freq,current_FCz_theta_compare,lag)[0,1]) for lag in range(-10,11)]

                    corr_list.append(",".join(rs))
                    freq_list.append(freq_index)
                    subject_list.append(sub_index)
                    node_region_list.append(node_index)
                    theta_list.append(theta_freq)

    # Time lagged cross correlation (TLCC) can identify directionality between two signals such as 
    # a leader-follower relationship in which the leader initiates a response which is repeated by the follower.
    # (does not identify causality)
    with open('eeg_correlations2.csv', mode='w', newline='') as eeg_file:
        eeg_writer = csv.writer(eeg_file, delimiter=',')

        eeg_writer.writerow(['node_region_filename', 'subject_index', 'subject_freq', 'theta_freq', 'tlcc'])

        val1 = np.array(eeg_filenames)[np.array(node_region_list)]
        val2 = np.array(subject_list)
        val3 = np.array(freq_list)
        val4 = np.array(theta_list)
        val5 = np.array(corr_list)

        rows = zip(val1,val2,val3,val4,val5)

        eeg_writer.writerows(rows)

def multivariate_data(dataset, target, start_index, history_size, target_size, step, end_index=None, single_step=False):
    '''
        target_size (int): how many timesteps into the future to predict
        step (int): defines time delay in lags (ex. step=1:1,2,3,4: step=2:1,3,5,7)
        single_step (bool): predicting one timestep or multiple
    '''
    data = []
    labels = []

    # get start and end indices
    start_index = start_index + history_size
    if end_index is None:
        end_index = dataset.shape[-1] - target_size

    # generate samples
    for i in range(start_index, end_index):

        # get input sample index range
        indices = range(i-history_size, i, step)

        # get input sample
        data.append(dataset[:,indices])

        # get label
        # TODO this can be changed to get rid of single_step variable
        if single_step:
            labels.append(target[:,i+target_size-1])
        else:
            labels.append(target[:,i:i+target_size-1])

    # return np.array(data), np.array(labels)
    return data, labels

def convert_dataset():

    
    # get target dataset
    DATA_FOLDER_PATH = 'D:\power_out2\LNR\evoked'
    FCz_data = scipy.io.loadmat(DATA_FOLDER_PATH + "/" + "POWevoked_leftnorew_FCz.mat")
    feedback_onset_FCz_theta =  FCz_data['POW_evoked'][:,3:7,626:888]
    stationary_FCz_theta = np.empty((feedback_onset_FCz_theta.shape[0],feedback_onset_FCz_theta.shape[1],260))
    for i in range(feedback_onset_FCz_theta.shape[0]):
        for j in range(feedback_onset_FCz_theta.shape[1]):
            current_FCz_theta_compare = feedback_onset_FCz_theta[i,j]
            current_FCz_theta_compare,_,_ = make_stationary(current_FCz_theta_compare)
            stationary_FCz_theta[i,j] = current_FCz_theta_compare
   

    # get stationary input dataset
    input_stationary_data = []
    input_region_filenames = ["POWevoked_leftnorew_FC1.mat","POWevoked_leftnorew_FC2.mat"]
    for filename in input_region_filenames:
        temp_data = scipy.io.loadmat(DATA_FOLDER_PATH + "/" + filename)
        temp_theta_data =  temp_data['POW_evoked'][:,3:7,626:888]
        stationary_temp_theta = np.empty((temp_theta_data.shape[0],temp_theta_data.shape[1],260))

        for i in range(temp_theta_data.shape[0]):
            for j in range(temp_theta_data.shape[1]):
                current_theta = temp_theta_data[i,j]
                current_theta,_,_ = make_stationary(current_theta)
                stationary_temp_theta[i,j] = current_theta
        
        input_stationary_data.append(np.copy(stationary_temp_theta))
    
    input_stationary_data = np.concatenate((input_stationary_data[0],input_stationary_data[1]),axis=-2)
    print(input_stationary_data.shape)
    
    # generate samples
    # input_stationary_data = (130,8,260)
    # stationary_FCz_theta  = (130,4,260)
    random.seed(10)
    temp_indices = range(130)
    training_indices = random.sample(temp_indices, 104)
    training_data = []
    training_labels = []
    for i in training_indices:
        # current_data = (num_samples,8,num_lags)
        # current_labels = (num_samples,4)
        current_data,current_labels = multivariate_data(input_stationary_data[i],stationary_FCz_theta[i],start_index=0,history_size=5,target_size=1,step=1,single_step=True)
        training_data += current_data
        training_labels += current_labels
    training_data = np.array(training_data)
    training_labels = np.array(training_labels)


    testing_indices = [x for x in temp_indices if x not in training_indices]
    testing_data = []
    testing_labels = []
    for i in testing_indices:
        # current_data = (num_samples,8,num_lags)
        # current_labels = (num_samples,4)
        current_data,current_labels = multivariate_data(input_stationary_data[i],stationary_FCz_theta[i],start_index=0,history_size=5,target_size=1,step=1,single_step=True)
        testing_data += current_data
        testing_labels += current_labels
    testing_data = np.array(testing_data)
    testing_labels = np.array(testing_labels)

    print(training_data.shape)
    print(training_labels.shape)
    print(testing_data.shape)
    print(testing_labels.shape)

    training_data = np.transpose(training_data,[0,2,1])
    testing_data = np.transpose(testing_data,[0,2,1])

    np.savez('eeg_dataset_training2', x=training_data, y=training_labels)
    np.savez('eeg_dataset_testing2', x=testing_data, y=testing_labels)

def load_dataset(dataset_filepath):
    dataset = np.load(dataset_filepath)

    return dataset

if __name__ == "__main__":
    convert_dataset()
