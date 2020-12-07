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

clean_print = (eeg_data[0])
del clean_print['Markers']
del clean_print['Channels']
del clean_print['ChannelCount']
del clean_print['MarkerCount']
del clean_print['SampleRate']
del clean_print['SegmentCount']
del clean_print['__header__']
del clean_print['__version__']
del clean_print['__globals__']

print(clean_print)