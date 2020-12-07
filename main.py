import scipy.io
import os

DATA_FOLDER_PATH = '/home/matt/eeg_data/Export File'

eeg_data = []
counter = 0
for f in os.listdir(DATA_FOLDER_PATH):
    if counter == 1:
        break
    eeg_data.append(scipy.io.loadmat(DATA_FOLDER_PATH + "/" + f))
    counter += 1

print(eeg_data)