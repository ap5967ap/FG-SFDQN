import os
import numpy as np
import torch
import ast
import configparser
from tasks.reacher import Reacher
from features.deep_fg import DeepFGSF
from agents.fgsfdqn import FGSFDQN
from agents.buffer import ConditionalReplayBuffer, ReplayBuffer

def load_config(path="configs/reacher.cfg"):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg

def main():
    config_params = load_config()
    cfg = config_params

    gen_params = config_params['GENERAL']
    n_samples = int(gen_params['n_samples'])

    task_params = config_params['TASK']
    goals = ast.literal_eval(task_params['train_targets'])
    test_goals = ast.literal_eval(task_params['test_targets'])
    all_goals = goals + test_goals

    def generate_tasks(include_target):
        train_tasks = [Reacher(all_goals, i, include_target) for i in range(len(goals))]
        test_tasks = [Reacher(all_goals, i + len(goals), include_target) for i in range(len(test_goals))]
        return train_tasks, test_tasks

    train_tasks, test_tasks = generate_tasks(True)

    input_dim = train_tasks[0].encode_dim()
    n_actions = train_tasks[0].action_count()
    n_features = train_tasks[0].feature_dim()

    lr_sf       = float(cfg["SFQL"]["learning_rate"])
    lr_w        = float(cfg["SFQL"]["learning_rate_w"])
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

    buffer_size = int(cfg["GENERAL"].get("buffer_size"))
    n_batch     = int(cfg["GENERAL"]["n_batch"])
    buffer = ConditionalReplayBuffer(
        n_samples=buffer_size,
        n_batch=n_batch
    )

    gamma       = float(cfg["AGENT"]["gamma"])
    epsilon     = float(cfg["AGENT"]["epsilon"])
    T           = int(cfg["AGENT"]["T"])
    algorithm   = "agl1"
    n_averaging = int(cfg["FGSF"].get("n_averaging",5))

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
        
        
    if algorithm != "alg1":
            agent.train_randomized(
                train_tasks=train_tasks,
                n_total_steps=n_samples * len(train_tasks),
                viewers=None
            )
    else:
        agent.train(train_tasks,n_samples=n_samples)
    return (agent,train_tasks,test_tasks)
    