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
import pandas as pds
import jsons

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

    DATA_FOLDER_PATH = '/home/matt/eeg_data/power/RR/evoked'
    FCz_data = scipy.io.loadmat(DATA_FOLDER_PATH + "/" + "POWevoked_rightrew_FCz.mat")
    print(FCz_data.keys())
    # print(FCz_data)
    eeg_data = []
    eeg_filenames = []
    counter = 0
    for f in os.listdir(DATA_FOLDER_PATH):

        if 'FCz' in f or 'topomaps' in f or 'workspace' in f:
            continue
        elif counter <= 29:
            pass
        else:
            print('Recording Filenames')
            print(f)
            eeg_filenames.append(f)
            eeg_data.append(scipy.io.loadmat(DATA_FOLDER_PATH + "/" + f))
            
        if counter == 39:
            break
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
        print(node_region.keys())
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
    with open('eeg_correlations_RRevoked4.csv', mode='w', newline='') as eeg_file:
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

def crop_tlcc_data(csv_file):
    tlcc_max_list = []
    time_lag_list = []
    df = pds.read_csv(csv_file)
    tlcc_df = df['tlcc'].to_list()
    for l in tlcc_df:
        tlcc_vals = [float(val) for val in l.split(",")]
        if max(tlcc_vals) >= abs(min(tlcc_vals)):
            tlcc_max_list.append(max(tlcc_vals))
            time_lag_list.append(tlcc_vals.index(max(tlcc_vals)))
        else:
            tlcc_max_list.append(min(tlcc_vals))
            time_lag_list.append(tlcc_vals.index(min(tlcc_vals)))
    df['tlcc'] = tlcc_max_list
    df['lag'] = time_lag_list
    df = find_best_theta(df)
    df.to_csv('csvs/eeg_correlations_RRevoked4_cropped.csv')

def find_best_theta(df):
    max_theta_df_list = []
    for i in range(0, len(df.index), 4):
        tlcc_list = [abs(df.loc[i]['tlcc']),abs(df.loc[i+1]['tlcc']),abs(df.loc[i+2]['tlcc']),abs(df.loc[i+3]['tlcc'])]
        max_theta = tlcc_list.index(max(tlcc_list))
        max_theta_df_list.append(df.loc[[i+max_theta]])
    new_df = pds.concat(max_theta_df_list)
    return new_df

def convert_tlcc(df):
    tlcc_df = df['tlcc'].to_list()
    tlcc_list = []
    for l in tlcc_df:
        tlcc_vals = [float(val) for val in l.split(",")]
        tlcc_list.append(tlcc_vals)
    df['tlcc'] = tlcc_list
    return df

def most_sig_freq(df):
    file_list = []
    subject_list = []
    subject_freq = []
    theta_list = []
    tlcc_list = []
    lag_list = []
    for idx, row in df.iterrows():
        if idx == 0:
            sig_freq = row
        elif row['subject_freq']/120 == 0 and idx != 0:
            file_list.append(sig_freq['node_region_filename'])
            subject_list.append(sig_freq['subject_index'])
            subject_freq.append(sig_freq['subject_freq'])
            theta_list.append(sig_freq['theta_freq'])
            tlcc_list.append(sig_freq['tlcc'])
            lag_list.append(sig_freq['lag'])
            sig_freq = row
        elif abs(row['tlcc']) > abs(sig_freq['tlcc']):
            sig_freq = row
        else:
            continue
    data_f = pds.DataFrame(
        {'file': file_list,
        'subject_id': subject_list,
        'subject_freq': subject_freq,
        'theta_freq': theta_list,
        'tlcc': tlcc_list,
        'lag': lag_list
        }
    )
    return data_f

def concat_csvs(df1, df2):
    final_df = pds.concat([df1, df2])
    return final_df.to_csv('csvs/combo_mean_eeg_correlations_RRevoked.csv', index=None)

def save_np_out(df):
    tlcc_list = []
    np_df = df.to_numpy()
    # print(np_df.shape)
    for i in range(np_df.shape[0]):
        test_arr = np.asarray(jsons.loads(np_df[i,4]))
        # tlcc_list.append(np.mean(np.absolute(test_arr)))
        tlcc_list.append(test_arr)
    tlcc_arr = np.vstack(tlcc_list)
    np_df = np.delete(np_df, 4, 1)
    np_df = np.hstack((np_df, tlcc_arr))
    # np_df[:,4] = np.asarray(tlcc_list)
    # np_df = np_df.reshape((-1,135,120,4,1))
    # print(np_df[0,:])
    # print(np_df.shape)
    np.save('numpys/eeg_correlations_RRevoked4.npy', np_df)

def load_np(data):
    mean_list = []
    data = data.reshape(-1,120*4,5)
    subject_id_total = []
    file_list = data[:,0,0].tolist()
    print(data.shape[0])
    for i in range(data.shape[0]):
        subject_id_total.append(i%135)
        mean_list.append(np.mean(np.absolute(data[i,:,4])))
    print(len(mean_list))
    print(len(subject_id_total))
    print(len(file_list))
    data_f = pds.DataFrame(
        {'file': file_list,
        'subject_id': subject_id_total,
        'means': mean_list
        }
    )
    data_f.to_csv('csvs/eeg_correlations_RRevoked4_means.csv', index=None)

def graph(data_file):
    data = pds.read_csv(data_file, index_col=1)
    print(data)
    data.plot(kind='bar')
    plt.ylabel('Averages')
    plt.xlabel('Files')
    plt.title('Title')
    plt.show()

if __name__ == "__main__":
    # convert_dataset()
    # get_corr_data()
    # crop_tlcc_data('csvs/eeg_correlations_RRevoked4.csv')
    #avg_freq(pds.read_csv('csvs/eeg_correlations_LNRevoked1.csv'))#.to_csv('csvs/eeg_correlations_LNRevoked1_avg.csv', index=None)
    # convert_tlcc(pds.read_csv('csvs/eeg_correlations_RRevoked4.csv')).to_csv('csvs/eeg_correlations_RRevoked4.csv', index=None)
    save_np_out(pds.read_csv('csvs/eeg_correlations_RRevoked4.csv'))
    # concat_csvs(pds.read_csv('csvs/combo_mean_eeg_correlations_RRevoked.csv'), pds.read_csv('csvs/eeg_correlations_RRevoked4_means.csv'))
    # load_np(np.load('numpys/eeg_correlations_RRevoked4.npy', allow_pickle=True))
    # graph('csvs/combo_mean_eeg_correlations_LNRevoked.csv')
