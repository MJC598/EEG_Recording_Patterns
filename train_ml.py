import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import scipy.io
import random
import pandas as pds

from scipy import stats
from sklearn.metrics import r2_score

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
        self.rnn = nn.GRU(input_size,hidden_size,num_layers=1,batch_first=batch_first,dropout=dropout)
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
    print(data.keys())
    print(data['x'].shape)
    x = data['x'].reshape((134, -1, 251))
    y = data['WSLS']
    print(x.shape)
    print(y.shape)
    # print(x[0,0,0,0,0,:])
    random.seed(10)
    full_indices = range(134)
    training_indices = random.sample(full_indices, k=90)
    for i in full_indices:
        if i in training_indices:
            training_data.append(x[i,:,:])
            training_labels.append(y[i,0])
        else:
            testing_data.append(x[i,:,:])
            testing_labels.append(y[i,0])
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
    for epoch in range(20):

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
        # print(np.squeeze(np.transpose(output.detach().cpu().numpy())))
        # print(y.detach().cpu().numpy())
        # print(stats.spearmanr(np.squeeze(np.transpose(output.detach().cpu().numpy())), y.detach().cpu().numpy()))
        # print(r2_score(y.detach().cpu().numpy(), np.squeeze(np.transpose(output.detach().cpu().numpy()))))
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
    loss_df.to_csv('LSTM1_loss_scores_wsls.csv', index=None)
    torch.save(model, save_filepath)

def r2_score_eval(model, testing_dataloader):
    output_list = []
    labels_list = []
    for i, (x, y) in enumerate(testing_dataloader):       
        x = x.permute(0, 2, 1)
        # print(x.shape)
        output = model(x) 
        output_list.append(np.squeeze(np.transpose(output.detach().cpu().numpy())))
        labels_list.append(y.detach().cpu().numpy())
    output_list = np.hstack(output_list)
    labels_list = np.hstack(labels_list)
    # print(np.hstack(output_list))
    # print(np.hstack(labels_list))
    print(r2_score(labels_list, output_list))

if __name__ == "__main__":
    input_size = 60
    hidden_size = 251
    batch_first = True
    batch_size = 2
    model = baselineLSTM(input_size,hidden_size,batch_size,batch_first,0)
    # model = baselineGRU(input_size,hidden_size,batch_size,batch_first,0)
    # model = baselineRNN(input_size,hidden_size,batch_size,batch_first)
    # model = baselineFCNLSTM(input_size,hidden_size,batch_size,batch_first)
    #['x', 'y', 'subject_id', 'channel', 'reward_type', 'power_type', 'frequency', 'time']
    # training_dataset, validation_dataset = get_data_from_mat('matlab/FCz2latency.mat')
    #['RT', 'WSLS', 'x', 'subject_id', 'channels', 'rew_type', 'pow_type', 'freq_bands', 't_window', 'fs']
    #input: subj, channel, rew_type, power, freq, time
    #output: mean reward, mean no reward, difference, std reward, std no reward
    training_dataset, validation_dataset = get_data_from_mat('matlab/FCzC3C42WSLS.mat')
    # training_dataset = get_dataset('eeg_dataset_training2.npz')
    training_loader = DataLoader(dataset=training_dataset,batch_size=batch_size,shuffle=True)

    # validation_dataset = get_dataset('eeg_dataset_testing2.npz')
    validation_loader = DataLoader(dataset=validation_dataset,batch_size=batch_size)

    # PATH = 'baselineLSTM.pth'
    # PATH = 'baselineGRU.pth'
    PATH = 'baselineLSTM_full_wsls.pth'
    # PATH = 'baselineFCNLSTM.pth'
    train_model(model,PATH,training_loader,validation_loader)
    model = torch.load(PATH)
    model.eval()
    r2_score_eval(model, training_loader)
    r2_score_eval(model, validation_loader)
