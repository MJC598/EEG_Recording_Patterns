import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import scipy.io
import random
import pandas as pds

class baselineRNN(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,batch_first):
        super(baselineRNN, self).__init__()
        self.rnn1 = nn.RNN(input_size,hidden_size,batch_first=batch_first,dropout=0.5)
        self.lin = nn.Linear(hidden_size,1)
        self.h0 = torch.randn(1, batch_size, hidden_size)

    def forward(self, x):
        x, h_n  = self.rnn1(x,self.h0)

        # take last cell output
        out = self.lin(x[:, -1, :])

        return out

class baselineLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,batch_first,dropout=0.5):
        super(baselineLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size,hidden_size,batch_first=batch_first,dropout=dropout)
        self.lin = nn.Linear(hidden_size,1)
        self.h0 = torch.randn(1, batch_size, hidden_size)
        self.c0 = torch.randn(1, batch_size, hidden_size)

    def forward(self, x):
        x, (h_n, c_n)  = self.rnn(x,(self.h0,self.c0))

        # take last cell output
        out = self.lin(x[:, -1, :])

        return out

class baselineGRU(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,batch_first,dropout=0.5):
        super(baselineGRU, self).__init__()
        self.rnn = nn.GRU(input_size,hidden_size,batch_first=batch_first,dropout=dropout)
        self.lin = nn.Linear(hidden_size,1)
        self.h0 = torch.randn(1, batch_size, hidden_size)

    def forward(self, x):
        # print(self.h0.shape)
        x, h_n  = self.rnn(x,self.h0)

        # take last cell output
        out = self.lin(x[:, -1, :])

        return out

class baselineFCNLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,batch_first):
        super(baselineFCNLSTM, self).__init__()
        self.conv1d_l1 = nn.Conv1d(128,256,8,stride=1)
        self.conv1d_l2 = nn.Conv1d(256,128,5,stride=1)
        self.conv1d_l3 = nn.Conv1d(128,128,3,stride=1)
        self.batchNorm_l1 = nn.BatchNorm1d(256, eps=0.001, momentum=0.99)
        self.batchNorm_l2 = nn.BatchNorm1d(128, eps=0.001, momentum=0.99)
        self.relu = nn.ReLU()
        self.pooling = nn.AvgPool1d(input_size)
        self.lstm = nn.LSTM(input_size,hidden_size,batch_first=batch_first)
        self.h0 = torch.randn(1, batch_size, hidden_size)
        self.c0 = torch.randn(1, batch_size, hidden_size)
        self.sm = nn.Softmax()

    def forward(self, x):
        y, (h_n, c_n)  = self.lstm(x,(self.h0,self.c0))
        x = self.relu(self.batchNorm_l1(self.conv1d_l1(x)))
        x = self.relu(self.batchNorm_l2(self.conv1d_l2(x)))
        x = self.relu(self.batchNorm_l2(self.conv1d_l3(x)))
        x = self.pooling(x)
        out = self.sm(torch.cat(x, y))
        return out

def get_dataset(data_filepath):
    data = np.load(data_filepath)

    tensor_x = torch.Tensor(data['x'])
    tensor_y = torch.Tensor(data['y'])

    dataset = TensorDataset(tensor_x,tensor_y)

    return dataset

def get_data_from_mat(data_filepath):
    training_data = []
    training_labels = []
    testing_data = []
    testing_labels = []
    data = scipy.io.loadmat(data_filepath)
    # print(data.keys())
    x = data['x']
    y = data['y']
    # print(x.shape)
    # print(x[0,0,0,0,0,:])
    random.seed(10)
    full_indices = range(134)
    training_indices = random.sample(full_indices, k=90)
    for i in full_indices:
        if i in training_indices:
            training_data.append(x[i,0,:,0,0,:])
            training_labels.append(y[i,2])
        else:
            testing_data.append(x[i,0,:,0,0,:])
            testing_labels.append(y[i,2])
    print(np.array(training_data).shape)
    # print(x.shape)
    # print(y.shape)
    training_dataset = TensorDataset(torch.Tensor(np.array(training_data)), torch.Tensor(np.array(training_labels)))
    testing_dataset = TensorDataset(torch.Tensor(np.array(testing_data)), torch.Tensor(np.array(testing_labels)))
    return training_dataset, testing_dataset


def train_model(model,save_filepath,training_loader,validation_loader):
    
    epochs_list = []
    train_loss_list = []
    val_loss_list = []
    training_len = len(training_loader.dataset)
    validation_len = len(validation_loader.dataset)

    data_loaders = {"train": training_loader, "val": validation_loader}

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss()

    # training and testing
    for epoch in range(1000):

        train_loss = 0.0
        val_loss = 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            for i, (x, y) in enumerate(data_loaders[phase]):       
                x = x.permute(0, 2, 1)
                # print(x.shape)
                output = model(x)                              
                loss = loss_func(torch.squeeze(output), torch.squeeze(y))                  
                optimizer.zero_grad()           

                if phase == 'train':
                    loss.backward()
                    optimizer.step()                                      

                running_loss += loss.item()
            
            if phase == 'train':
                train_loss = running_loss
            else:
                val_loss = running_loss

        # shows average loss
        # print('[%d, %5d] train loss: %.6f val loss: %.6f' % (epoch + 1, i + 1, train_loss/training_len, val_loss/validation_len))
        # shows total loss
        print('[%d, %5d] train loss: %.6f val loss: %.6f' % (epoch + 1, i + 1, train_loss, val_loss))
        epochs_list.append(epoch)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    loss_df = pds.DataFrame(
        {
            'epoch': epochs_list,
            'training loss': train_loss_list,
            'validation loss': val_loss_list
        }
    )
    loss_df.to_csv('loss_scores.csv')
    torch.save(model, save_filepath)

if __name__ == "__main__":
    input_size = 2
    hidden_size = 64
    batch_first = True
    batch_size = 2
    # model = baselineLSTM(input_size,hidden_size,batch_size,batch_first)
    model = baselineGRU(input_size,hidden_size,batch_size,batch_first)
    # model = baselineRNN(input_size,hidden_size,batch_size,batch_first)
    # model = baselineFCNLSTM(input_size,hidden_size,batch_size,batch_first)

    training_dataset, validation_dataset = get_data_from_mat('matlab/FCz2latency.mat')
    # training_dataset = get_dataset('eeg_dataset_training2.npz')
    training_loader = DataLoader(dataset=training_dataset,batch_size=batch_size,shuffle=True)

    # validation_dataset = get_dataset('eeg_dataset_testing2.npz')
    validation_loader = DataLoader(dataset=validation_dataset,batch_size=batch_size)

    # PATH = 'baselineLSTM.pth'
    # PATH = 'baselineGRU.pth'
    PATH = 'baselineGRU_Theta2Mean_Dropout.pth'
    # PATH = 'baselineFCNLSTM.pth'
    train_model(model,PATH,training_loader,validation_loader)
