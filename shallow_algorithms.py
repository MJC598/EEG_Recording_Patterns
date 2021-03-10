import numpy as np
import scipy.io
import random
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

def get_data_from_mat(data_filepath, index, second_data=None):
    training_data = []
    training_labels = []
    testing_data = []
    testing_labels = []
    data = scipy.io.loadmat(data_filepath)
    # print(data.keys())
    # print(data['Cond'][0,0].shape)
    # print(data['Cond'][0,0])
    # print(data['behav'].shape)
    # print(data['X'][0,1].shape)
    # x = data['X'].reshape((134, -1, 251))
    # x = data['X'][0,0].reshape((12562, -1, 251))
    # x = data['X'][0,1][:,0,1,25:150].reshape((12035, -1, 125))
    x = data['RP_top_ft']
    x2 = (data['RP_ft_top_i'][:,index]) - 1
    # print(x.shape)
    # print(x2)
    y = data['behav'][:,index]
    # print(y.shape)
    # print(data['RT_long'][0,1].shape)
    # y = np.squeeze(data['LR'][0,1])
    # print(y)
    # y_oh = np.zeros((y.size, y.max()+1))
    # y_oh[np.arange(y.size),y] = 1
    # y = y_oh
    # print(y_oh)
    # print(x.shape)
    # print(y.shape)
    # print(x[0,0,0,0,0,:])
    random.seed(10)
    full_indices = range(134)
    training_indices = random.sample(full_indices, k=104)
    for i in full_indices:
        if i in training_indices:
            training_data.append(x[i,x2])
            training_labels.append(y[i])
            # print(y[i])
        else:
            testing_data.append(x[i,x2])
            testing_labels.append(y[i])
    # print(np.array(training_data).shape)
    # print(np.array(testing_data)[:1536,:,:].shape)
    # print(np.array(testing_labels).shape)
    # print(x.shape)
    # print(y.shape)
    # print(training_labels)
    training_dataset = (np.array(training_data), np.array(training_labels))
    testing_dataset = (np.array(testing_data), np.array(testing_labels))
    return training_dataset, testing_dataset

if __name__ == '__main__':
    input_file = 'matlab/RP_Behav_data.mat'
    # label_file = 'matlab/FCzC3C42WSLS'
    for index in range(7):
        training, testing = get_data_from_mat(input_file, index)
        linear_reg = LinearRegression().fit(training[0], training[1])
        kernel_reg = KernelRidge(kernel='poly', gamma=0.1).fit(training[0], training[1])
        print('-------------------------------------------------------------------------------------------------')
        lr_pred = linear_reg.predict((testing[0]))
        kr_pred = kernel_reg.predict((testing[0]))
        print('Index: {} Linear Regression Training: {}'.format(index, linear_reg.score(training[0], training[1])))
        print('Index: {} Linear Regression Testing: {}'.format(index, linear_reg.score(testing[0], testing[1])))
        print('Index: {} Kernel Regression Training: {}'.format(index, kernel_reg.score(training[0], training[1])))
        print('Index: {} Kernel Regression Testing: {}'.format(index, kernel_reg.score(testing[0], testing[1])))
    # plt.scatter(training[0], training[1], color='green')
    # plt.scatter(testing[0], testing[1], color='blue')
    # plt.plot(testing[0], lr_pred, color='red', linewidth=3)
    # plt.plot(testing[0], kr_pred, color='black', linewidth=3)
    # plt.show()