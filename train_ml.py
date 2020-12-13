import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class myRNN(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,batch_first):
        super(myRNN, self).__init__()
        self.rnn1 = nn.RNN(input_size,hidden_size,batch_first=batch_first)
        self.lin = nn.Linear(hidden_size,4)
        self.h0 = torch.randn(1, batch_size, hidden_size)

    def forward(self, x):
        x, h_n  = self.rnn1(x,self.h0)

        # take last cell output
        out = self.lin(x[:, -1, :])

        return out

class myLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,batch_first):
        super(myLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size,hidden_size,batch_first=batch_first)
        self.lin = nn.Linear(hidden_size,4)
        self.h0 = torch.randn(1, batch_size, hidden_size)
        self.c0 = torch.randn(1, batch_size, hidden_size)

    def forward(self, x):
        x, (h_n, c_n)  = self.rnn(x,(self.h0,self.c0))

        # take last cell output
        out = self.lin(x[:, -1, :])

        return out

class myGRU(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,batch_first):
        super(myGRU, self).__init__()
        self.rnn = nn.GRU(input_size,hidden_size,batch_first=batch_first)
        self.lin = nn.Linear(hidden_size,4)
        self.h0 = torch.randn(1, batch_size, hidden_size)

    def forward(self, x):
        x, h_n  = self.rnn(x,self.h0)

        # take last cell output
        out = self.lin(x[:, -1, :])

        return out

def get_dataset(data_filepath):
    data = np.load(data_filepath)

    tensor_x = torch.Tensor(data['x'])
    tensor_y = torch.Tensor(data['y'])

    dataset = TensorDataset(tensor_x,tensor_y)

    return dataset

def train_model(model,training_loader,validation_loader):

    training_len = len(training_loader.dataset)
    validation_len = len(validation_loader.dataset)

    data_loaders = {"train": training_loader, "val": validation_loader}

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss()

    # training and testing
    for epoch in range(100):

        train_loss = 0.0
        val_loss = 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            for i, (x, y) in enumerate(data_loaders[phase]):       
    
                output = model(x)                              
                loss = loss_func(output, y)                  
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



if __name__ == "__main__":
    input_size = 8
    hidden_size = 64
    batch_first = True
    batch_size = 52
    model = myGRU(input_size,hidden_size,batch_size,batch_first)

    training_dataset = get_dataset('eeg_dataset_training2.npz')
    training_loader = DataLoader(dataset=training_dataset,batch_size=batch_size,shuffle=True)

    validation_dataset = get_dataset('eeg_dataset_testing2.npz')
    validation_loader = DataLoader(dataset=validation_dataset,batch_size=batch_size)

    train_model(model,training_loader,validation_loader)