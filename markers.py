import numpy as np
import scipy.io
import os, csv

marker_info = {'S 1':[], 'S 2':[], 'S 4':[], 'S 5':[], 'S 6':[], 'S 7':[], 'S 8':[], 'S 9':[]}

def get_markers():
    DATA_FOLDER_PATH = '/home/matt/eeg_data/Export File'
    test_file = scipy.io.loadmat(DATA_FOLDER_PATH + "/" + "3NBOA0001_left_NoReward.mat")
    print(test_file['Markers'])

if __name__ == "__main__":
    get_markers()