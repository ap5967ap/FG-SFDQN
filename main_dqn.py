import configparser
import ast
import os
import time
import numpy as np
from tasks.gridworld import Shapes
from agents.buffer import ReplayBuffer
from agents.dqn import DQN
import torch
import torch.nn as nn
import torch.optim as optim

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_cfg(path="configs/config.cfg"):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    conf = {}
    conf["n_samples"] = cfg.getint("GENERAL", "n_samples")
    conf["n_tasks"] = cfg.getint("GENERAL", "n_tasks")
    conf["n_trials"] = cfg.getint("GENERAL", "n_trials")
    conf["n_batch"] = cfg.getint("GENERAL", "n_batch")

    conf["maze"] = np.array(ast.literal_eval(cfg.get("TASK", "maze")),dtype=str)
    
    conf["gamma"] = cfg.getfloat("AGENT", "gamma")
    conf["epsilon"] = cfg.getfloat("AGENT", "epsilon")
    conf["T"] = cfg.getint("AGENT", "T")
    conf["print_ev"] = cfg.getint("AGENT", "print_ev")
    conf["save_ev"] = cfg.getint("AGENT", "save_ev")

    conf["ql_lr"] = cfg.getfloat("QL", "learning_rate")
    return conf


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate=1e-3):
        super(MLP, self).__init__()
        
        # --- 1. DEFINE the network here ---
        # Let's call it self.network
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        ).to(device)
        
        # --- 2. USE the SAME name here ---
        # The optimizer needs the parameters from the self.network instance
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        self.loss = nn.MSELoss()

    def forward(self, x):
        # --- 3. And USE it again here ---
        return self.network(x.to(device))




def main():
    
    def mlp_build():
        input_dim = task.encode_dim()
        output_dim = task.action_count()
        model = MLP(input_dim, output_dim, learning_rate=1e-3)
        return model
    conf = load_cfg("configs/config.cfg")
    maze = conf["maze"]
    task = Shapes(maze=maze,shape_rewards={'1':10,'2':0,'3':0})
    buffer = ReplayBuffer(n_samples=conf['n_samples'], n_batch=conf['n_batch']) 
    agent=DQN(mlp_build,buffer,gamma=conf["gamma"],epsilon=conf["epsilon"],T=conf["T"],encoding=task.encode)
    agent.reset()


    train_tasks = [task]
    n_samples=conf['n_samples']
    agent.train(train_tasks=train_tasks, n_samples=n_samples, test_tasks=[])
    return (agent,train_tasks)

