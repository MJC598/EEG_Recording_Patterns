import torch.nn as nn
import numpy as np
import torch
import train_ml
from train_ml import baselineGRU, baselineLSTM, baselineRNN
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def get_model(path):

    model = torch.load(path)
    model.eval()

    return model

def evaluate_model(my_model,test_loader,frequency,interval):

    values = list(test_loader)
    test_inputs = values[0][0]
    target_outputs = values[0][1]

    predictions = my_model(test_inputs)

    frequency -= 4
    plt.plot(target_outputs.detach().numpy()[:interval][:,frequency], color="blue", label='target')
    plt.plot(predictions.detach().numpy()[:interval][:,frequency], color="red", label='predictions')
    plt.legend(loc='best')
    plt.title('eeg theta predictions vs target values')
    plt.show()
        

if __name__ == "__main__":

    model_path = 'baselineRNN.pth'
    my_model = get_model(model_path)

    batch_size = 52
    test_dataset = train_ml.get_dataset('eeg_dataset_testing2.npz')
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size)

    training_dataset = train_ml.get_dataset('eeg_dataset_training2.npz')
    training_loader = DataLoader(dataset=training_dataset,batch_size=batch_size)

    frequency = 4
    interval = 20

    # evaluate generalization
    evaluate_model(my_model,test_loader,frequency,interval)

    # evaluate represenational capacity
    evaluate_model(my_model,training_loader,frequency,interval)