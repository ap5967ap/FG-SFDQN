import os
import numpy as np
import torch
import configparser
from tasks.gridworld import Shapes
from features.deep_fg import DeepFGSF
from agents.fgsfdqn import FGSFDQN
from agents.buffer import ConditionalReplayBuffer, ReplayBuffer


def load_config(path="configs/config.cfg"):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def setup_tasks(cfg):
    """
    Creates training and testing tasks with diverse reward structures.
    """
    maze = np.array(eval(cfg["TASK"]["maze"]))
    task_train_1 = Shapes(maze=maze, shape_rewards={'1': 10, '2': 1,  '3': 1})
    task_train_2 = Shapes(maze=maze, shape_rewards={'1': 1,  '2': 10, '3': 1})
    task_train_3 = Shapes(maze=maze, shape_rewards={'1': 1,  '2': 1,  '3': 10})
    
    task_train_4 = Shapes(maze=maze, shape_rewards={'1': 5, '2': 5, '3': 0})
    task_train_5 = Shapes(maze=maze, shape_rewards={'1': 0, '2': 5, '3': 5})
    task_train_6 = Shapes(maze=maze, shape_rewards={'1': 5, '2': 0, '3': 5})

    train_tasks = [task_train_1, task_train_2, task_train_3,task_train_4, task_train_5, task_train_6]

    test_task = Shapes(maze=maze, shape_rewards={'1': 10, '2': 10, '3': 10})
    
    return train_tasks, test_task

def main():
    cfg = load_config()
    n_samples = int(cfg["GENERAL"].get("n_samples", 20000))
    buffer_size = int(cfg["GENERAL"].get("buffer_size", 200000))
    gamma       = float(cfg["AGENT"]["gamma"])
    epsilon     = float(cfg["AGENT"]["epsilon"])
    T           = int(cfg["AGENT"]["T"])
    lr_sf       = float(cfg["SFQL"]["learning_rate"])
    lr_w        = float(cfg["SFQL"]["learning_rate_w"])
    algorithm   = cfg["FGSF"].get("algorithm", "alg3")
    n_averaging = int(cfg["FGSF"].get("n_averaging",5))
    n_batch     = int(cfg["GENERAL"]["n_batch"])
    
    train_tasks, test_task = setup_tasks(cfg)
    input_dim = train_tasks[0].encode_dim()
    n_actions = train_tasks[0].action_count()
    n_features = train_tasks[0].feature_dim()
    
    sf = DeepFGSF(
        input_dim=input_dim,
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=lr_sf,
        learning_rate_w=lr_w,
        target_update_ev=1000,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_true_reward=cfg["SFQL"].getboolean("use_true_reward", False)
    )
    buffer = None
    if algorithm == "alg3":
        buffer = ConditionalReplayBuffer(
            n_samples=buffer_size,
            n_batch=n_batch
        )
    else:
        buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
        
    
    agent = FGSFDQN(
        deep_sf=sf,
        buffer=buffer,
        gamma=gamma,
        T=T,
        epsilon=epsilon,
        epsilon_decay=0.99995, 
        epsilon_min=0.05,
        encoding="task",
        algorithm=algorithm,      
        n_averaging=n_averaging,  
        print_ev=2000,
        save_ev=200,
        use_gpi=True
    )
    
    agent.reset()
    
    if algorithm != "alg1":
        agent.train_randomized(
            train_tasks=train_tasks,
            n_total_steps=n_samples * len(train_tasks),
            viewers=None
        )
    else:
        agent.train(train_tasks,n_samples=n_samples)
        
    return (agent,train_tasks,test_task)