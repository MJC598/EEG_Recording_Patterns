import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import operator
import seaborn as sns
from collections import Counter

df = pd.read_csv('eeg_correlations.csv')
vals = df.values
correlations = vals[:,-1]
corr_list_sorted_indeces = np.argsort(correlations)
vals = vals[corr_list_sorted_indeces]
vals_bottom = vals[:100]
vals_top = vals[-100:]


original = plt.hist2d(vals_top[:,3]+4,vals_top[:,2]+1,bins=10)
plt.colorbar(original[3])
plt.title('FCz theta most Positive correlation frequency pairs')
plt.xlabel('FCz theta freq', fontsize=12)
plt.ylabel('subject freq', fontsize=12)
plt.show()

original = plt.hist2d(vals_bottom[:,3]+4,vals_bottom[:,2]+1,bins=10)
plt.colorbar(original[3])
plt.title('FCz theta most Negative correlation frequency pairs')
plt.xlabel('FCz theta freq', fontsize=12)
plt.ylabel('subject freq', fontsize=12)
plt.show()


# TODO add y label
subject_freq = vals_top[:,0]
suject_freq_count = Counter(subject_freq)
region_list = []
for x in suject_freq_count.keys():
    x = x.split("_")[-1]
    x = x[:-4]
    region_list.append(x)
original = plt.bar(region_list,suject_freq_count.values())
plt.title('regions with most Positive correlation to FCz thetaband')
plt.xlabel('node region', fontsize=12)
plt.show()

vals_bottom = vals[:20]
subject_freq = vals_bottom[:,0]
suject_freq_count = Counter(subject_freq)
region_list = []
for x in suject_freq_count.keys():
    x = x.split("_")[-1]
    x = x[:-4]
    region_list.append(x)
original = plt.bar(region_list,suject_freq_count.values())
plt.title('regions with most Negative correlation to FCz thetaband')
plt.xlabel('node region', fontsize=12)
plt.show()

# TODO look at each person and each person's regions
vals = vals[vals[:,1]==3]
vals = vals[vals[:,0]=="POWevoked_leftnorew_T7.mat"]
vals = np.array(sorted(vals,key=operator.itemgetter(2,3)))
vals = vals[:,-1]
reshape_len = int(vals.shape[-1] / 4)
vals = vals.reshape((reshape_len,4)).astype(np.float32)

sns.set()
ax = sns.heatmap(vals, vmin=-1, vmax=1,cmap='bwr')
plt.xlabel('FCz theta frequencies', fontsize=12)
plt.ylabel('FC1 frequencies', fontsize=12)
plt.show()