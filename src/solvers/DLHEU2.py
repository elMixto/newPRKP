from src.data_structures import Instance
from src.data_structures.features import *
from src.solvers.collection import SolverCollection
import torch
from torch import nn
import os
from pathlib import Path
import sys





def encode_bool_vec(a):
    indices = a.nonzero()
    a-= 1
    indices.T[0]
    a[indices] = 1
    return a

def decode_bool_vec(a):
    a = a + 1
    return a/2

class Net(nn.Module):
    def __init__(self, entrada,salida):
        super(Net, self).__init__()

        hidden_size = 40

        self.many = nn.Sequential(
            nn.Linear(entrada, hidden_size),
            nn.Linear(hidden_size,hidden_size),
            nn.Linear(hidden_size,hidden_size),
            nn.Linear(hidden_size, salida),
            nn.Tanh()
        )

    def forward(self, x):   
        x = self.many(x)
        return x
    


class DHEU:
    def __init__(self,features: list[ItemBatchFeature| ItemSingleFeature]) -> None:
        self.net = Net(len(features),1)
        self.features = features
        self.criterion = nn.SmoothL1Loss()
        self.lr = 0.1
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr = self.lr)
        self.expand_vector = lambda x,instance: torch.cat([x,torch.zeros(max_size-instance.n_items)])
        self.get_y = lambda instance,sol: self.expand_vector(torch.tensor(sol),instance)
    
    def gen_x(self,instance: Instance):
        evaluated_features = []
        for feature in self.features:
            x_feature = torch.tensor(feature.batch_evaluate(instance))
            evaluated_features.append(x_feature)
        return torch.stack(evaluated_features)

    def train(self,training_instances_path: Path):
        instances = iter(map(Instance.from_file,training_instances_path.iterdir()))
        for epoch , instance in enumerate(instances):
            x = self.gen_x(instance).T
            y = encode_bool_vec(torch.tensor(instance.optimal_solution)).view(-1,1)
            self.optimizer.zero_grad()
            total_loss = 0
            y_pred = self.net(x)
            batch_loss = self.criterion(y_pred,y)
            batch_loss.backward()
            self.optimizer.step()
            total_loss += batch_loss.item()
            sys.stdout.write(f'\rEpoch {instance.n_items}, Loss {total_loss} y_pred {y_pred[0]}')
            sys.stdout.flush()

    def evaluate(self,instance):
        with torch.no_grad():
            x = self.gen_x(instance).T
            y = self.net(x)
            return decode_bool_vec(y).T[0]
        
    def save(self,path: Path):
        torch.save(self.net.state_dict(), path)
    
    def load(self,path: Path):
        self.net.load_state_dict(torch.load(path))
        #self.net.eval()

