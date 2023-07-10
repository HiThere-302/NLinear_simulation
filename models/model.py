import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import matplotlib.pyplot as plt

from collections import OrderedDict

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='best_model.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'best_model.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'weights/'+self.path)
        self.val_loss_min = val_loss

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)

        output, h_n = self.rnn(x, h_0)
        h_n = h_n.view(-1, 1, self.hidden_size)
        out = self.fc(h_n)

        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(LSTM, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size = input_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            batch_first = True)
        
        self.fc = nn.Sequential(nn.Dropout(0.5), 
                                nn.Linear(hidden_size, output_size)
                                )
        
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)

        output, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, 1, self.hidden_size)
        out = self.fc(h_out)

        return out
    
class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(NLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x # [Batch, Output length, Channel]

class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses= []
        self.val_losses = []
        
    def train_step(self, x, y):
        self.model.train()
        yhat = self.model(x)
        loss = self.loss_fn(y, yhat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        return loss.item()
    
    def train(self, train_loader, vali_loader, epochs=300):
        early_stopping = EarlyStopping(patience=20, verbose=True)

        for epoch in range(1, epochs+1):
            batch_losses = []
            for i, (train_x, train_y) in enumerate(train_loader):
                loss = self.train_step(train_x, train_y)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
                        
            with torch.no_grad():
                batch_val_losses = []
                for vali_x, vali_y in vali_loader:
                    self.model.eval()
                    yhat = self.model(vali_x)
                    val_loss = self.loss_fn(vali_y, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

                early_stopping(validation_loss, self.model)
            
                if early_stopping.early_stop:
                    print("----- Early stopping: Validation loss has stopped improving -----")
                    break
            if (epoch <= 10) | (epoch % 30 == 0) | (epoch == epochs):
                print(f"[{epoch}/{epochs}] Training loss: {training_loss:.4f}/t Validation loss: {validation_loss:.4f}")

    def predict(self, test_loader):
        with torch.no_grad():
            preds = []
            for test_x in test_loader:
                self.model.eval()
                pred = self.model(test_x)
                pred = pred.detach().cpu().numpy().flatten()
                for item in pred:
                    preds.append(item)

        return np.array(preds).flatten()
    
    def plot_train_history(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.train_losses, label='train loss')
        plt.plot(self.val_losses, label='validation loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
    
class RNN_Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.vali_losses = []

    def train_step(self, x, y):
        self.model.train()
        yhat = self.model(x)
        loss = self.loss_fn(yhat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def train(self, train_loader, vali_loader, epochs=300):
        early_stopping = EarlyStopping(patience=10, verbose=True)

        for epoch in range(1, epochs+1):
            batch_losses = []
            for i, (train_x, train_y) in enumerate(train_loader):
                loss = self.train_step(train_x, train_y)
                batch_losses.append(loss)
            current_train_loss = np.mean(batch_losses)
            self.train_losses.append(current_train_loss)

            with torch.no_grad():
                batch_vali_losses = []
                for vali_x, vali_y in vali_loader:
                    self.model.eval()
                    yhat = self.model(vali_x)
                    vali_loss = self.loss_fn(yhat, vali_y).item()
                    batch_vali_losses.append(vali_loss)
                current_vali_loss = np.mean(batch_vali_losses)
                self.vali_losses.append(current_vali_loss)

                early_stopping(current_vali_loss, self.model)
                if early_stopping.early_stop:
                    print("EARLY STOPPING")
                    print("VALIDATION LOSS HAS STOPPED IMPORVING")
                    break
            
            if (epoch <= 10) | (epoch % 30 ==0) | (epoch == epochs):
                print(f"[{epoch}/{epochs}] Training loss: {current_train_loss:.4f}/t Validation loss: {current_vali_loss:.4f}")
    
    def predict(self, test_loader):
        with torch.no_grad():
            preds = []
            for test_x in test_loader:
                self.model.eval()
                pred = self.model(test_x)
                pred = pred.detach().cpu().numpy().flatten()
                for item in pred:
                    preds.append(item)
        
        return np.array(preds).reshape(-1,1)
    
    def plot_train_history(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.train_losses, label='train loss')
        plt.plot(self.vali_losses, label='validation loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()


