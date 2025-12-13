import json
import os
import ast
import configparser
import numpy as np
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tasks.gridworld import Shapes
from features.deep import DeepSF
from features.deep_fg import DeepFGSF
from agents.buffer import ReplayBuffer, ConditionalReplayBuffer
from agents.dqn import DQN
from agents.sfdqn import SFDQN
from agents.fgsfdqn import FGSFDQN
from utils.config import *


config_params = load_config()
cfg = config_params
gen_params = config_params['GENERAL']
n_samples = int(gen_params['n_samples'])
agent_params = config_params['AGENT']
dqn_params = config_params['QL']
sfdqn_params = config_params['SFQL']
fgsfdqn_params = config_params['FGSF']

def build_task_sequence(cfg):
    maze = np.array(ast.literal_eval(cfg["TASK"]["maze"]), dtype=str)
    rewards_pool = [
        {'1': 1.0, '2': 0.0, '3': 0.0},
        {'1': 0.0, '2': 1.0, '3': 0.0},
        {'1': 0.0, '2': 0.0, '3': 1.0},
        {'1': 1.0, '2': -1.0, '3': 0.0},
        {'1': 0.0, '2': 1.0, '3': -1.0}
    ]
    
    n_tasks = int(cfg["GENERAL"]["n_tasks"])
    tasks = []
    for i in range(n_tasks):
        r = rewards_pool[i % len(rewards_pool)]
        tasks.append(Shapes(maze=maze, shape_rewards=r))
        
    return tasks
tasks = build_task_sequence(cfg)


def evaluate_agent(agent, tasks, n_episodes=10):
    results = []
    for i, task in enumerate(tasks):
        total_r = 0.0
        for _ in range(n_episodes):
            total_r += agent.test_agent(task, max_steps=100)
        avg_r = total_r / n_episodes
        results.append(avg_r)
    return results

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate=1e-3):
        super(MLP, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.network(x.to(self.device))

def make_agent(name, algo_type, cfg, input_dim, n_actions, n_features, encoding_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer_size = int(cfg["GENERAL"]["buffer_size"])
    n_batch = int(cfg["GENERAL"]["n_batch"])
    gamma = float(cfg["AGENT"]["gamma"])
    epsilon = float(cfg["AGENT"]["epsilon"])
    T = int(cfg["AGENT"]["T"])
    if name == "DQN":
        def model_builder():
            return MLP(input_dim, n_actions, learning_rate=float(cfg["QL"]["learning_rate"]))
        
        buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        return DQN(model_builder, buffer, gamma=gamma, epsilon=epsilon, T=T, encoding=tasks[0].encode, save_ev=1000)
    elif name == "SFDQN":
        sf = DeepSF(
                input_dim=input_dim, n_actions=n_actions, n_features=n_features,
                learning_rate=float(cfg["SFQL"]["learning_rate"]),
                learning_rate_w=float(cfg["SFQL"]["learning_rate_w"]),
                device=device, use_true_reward=cfg["SFQL"].getboolean("use_true_reward")
            )
        buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        return SFDQN(sf, buffer, gamma=gamma, epsilon=epsilon, T=T, encoding=encoding_fn, save_ev=1000)

    elif name == "FG-SFDQN":
        sf = DeepFGSF(
            input_dim=input_dim, n_actions=n_actions, n_features=n_features,
            learning_rate=float(cfg["SFQL"]["learning_rate"]),
            learning_rate_prior=float(cfg["SFQL"].get("learning_rate_prior", cfg["SFQL"]["learning_rate"])),
            learning_rate_w=float(cfg["SFQL"]["learning_rate_w"]),
            device=device, use_true_reward=cfg["SFQL"].getboolean("use_true_reward")
        )
        if algo_type == "alg3":
            buffer = ConditionalReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        else:
            buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
            
        return FGSFDQN(sf, buffer, gamma=gamma, epsilon=epsilon, T=T, encoding=encoding_fn, 
                       algorithm=algo_type, n_averaging=5, save_ev=1000)
    return None

n_samples = int(cfg["GENERAL"]["n_samples"])

input_dim = tasks[0].encode_dim()
n_actions = tasks[0].action_count()
n_features = tasks[0].feature_dim()

final_scores = {}

sequential_agents = [("DQN", None),("SFDQN", None),("FG-SFDQN", "alg1")]

for name, algo in sequential_agents:
    label = f"{name} ({algo if algo else 'Standard'})"
    print(f"Training {label} sequentially...")
    
    agent = make_agent(name, algo, cfg, input_dim, n_actions, n_features, tasks[0].encode)
    agent.reset()
    
    for i, task in enumerate(tasks):
        print(f"  > Task {i+1}...")
        agent.train_on_task(task, n_samples=n_samples)
        
    final_scores[label] = evaluate_agent(agent, tasks)
    print("  Done.")


randomized_agents = [("FG-SFDQN", "alg2"),("FG-SFDQN", "alg3")]

total_steps = n_samples * len(tasks)

for name, algo in randomized_agents:
    label = f"{name} ({algo})"
    print(f"Training {label} randomized...")
    
    agent = make_agent(name, algo, cfg, input_dim, n_actions, n_features, tasks[0].encode)
    agent.reset()
    
    agent.train_randomized(train_tasks=tasks, n_total_steps=total_steps)
    
    final_scores[label] = evaluate_agent(agent, tasks)
    print("  Done.")


for label, scores in final_scores.items():
    score_str = "".join([f" | {s:^10.1f}" for s in scores])
    print(f"{label:<25}{score_str}")


with open("final_scores_4room.json", "w") as f:
    json.dump(final_scores, f, indent=4)



