from math import ceil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
class Classify(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.seq = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, ceil(self.hidden_size/2)),
            nn.ReLU(inplace=True),
            nn.Linear(ceil(self.hidden_size/2), self.output_dim),
        )
        for layer in self.seq:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.1)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = x.view(-1, self.input_dim)
        x = self.seq(x)
        return x

class my_dataset(Dataset):
    def __init__(self, data, target):
        super().__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    
class Model:
    def __init__(self, input_dim, hidden_size, output_dim, episodes=80, lr=0.001, batch_size=32):
        self.model = Classify(input_dim, hidden_size, output_dim)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=1e-4)
        self.episodes = episodes
        self.batch_size = batch_size
    
    def predict(self, x:torch.Tensor):
        result = self.model(x)
        return torch.max(result, dim=1)[1]
    
    def train_and_test(self, data: np.ndarray, target: np.ndarray, data_test, target_test):
        data_all = torch.from_numpy(data).float()
        target_all = torch.from_numpy(target).long()
        dataset = my_dataset(data_all, target_all)
        loader = DataLoader(dataset)
        train_loss = []
        test_accuracy = []
        for episode in range(self.episodes):
            loss_sum = 0
            for data, target in loader:
                predict_prob = self.model(data)
                loss = self.loss_func(predict_prob, target)
                loss_sum += float(loss.detach().data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_loss.append(loss_sum/len(dataset))
            pred_all = self.predict(data_all).numpy()
            accuracy = accuracy_score(target_all.numpy(), pred_all)
            print('episode {} end, loss={:.6f}, accuracy={:.6f}'.format(episode+1, loss_sum/len(dataset), accuracy))
            test_acc = self.test(data_test, target_test)
            test_accuracy.append(test_acc)
        return train_loss, test_accuracy
    
    def test(self, data:np.ndarray, target:np.ndarray):
        data_test = torch.from_numpy(data).float()
        predict_result = self.predict(data_test).numpy()
        acc = accuracy_score(target, predict_result)
        print('test accuracy:{}'.format(acc))
        return acc

def plot(title:str, x_name:str, y_name:str, x, y):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    plt.show()
    plt.savefig('{}.svg'.format(title))