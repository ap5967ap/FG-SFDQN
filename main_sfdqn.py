import configparser
import numpy as np
import torch
import numpy as np
from tasks.gridworld import Shapes
from agents.sfdqn import SFDQN
from agents.buffer import ReplayBuffer
from features.deep import DeepSF

def load_config(path="configs/config.cfg"):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def build_tasks(cfg):
    maze = np.array(eval(cfg["TASK"]["maze"]))
    task_train_1 = Shapes(maze=maze, shape_rewards={'1': 10, '2': 1, '3': 2})
    task_train_2 = Shapes(maze=maze, shape_rewards={'1': 1, '2': 10, '3': 2})
    task_train_3 = Shapes(maze=maze, shape_rewards={'1': 1, '2': 2, '3': 10})
    test_task = Shapes(maze=maze, shape_rewards={'1': 10, '2': 10, '3': 10})
    return [task_train_1, task_train_2, task_train_3], test_task



def main(train_tasks=None, test_task=None):
    cfg = load_config()
    n_samples = int(cfg["GENERAL"]["n_samples"])
    n_batch   = int(cfg["GENERAL"]["n_batch"])
    gamma     = float(cfg["AGENT"]["gamma"])
    epsilon   = float(cfg["AGENT"]["epsilon"])
    T         = int(cfg["AGENT"]["T"])
    if train_tasks is None:
        train_tasks = build_tasks(cfg)[0]
    if test_task is None:
        test_task = build_tasks(cfg)[1]
    input_dim = train_tasks[0].encode_dim()
    n_actions = train_tasks[0].action_count()
    n_features = train_tasks[0].feature_dim()


    sf = DeepSF(
        input_dim=input_dim,
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=float(cfg["SFQL"]["learning_rate"]),
        learning_rate_w=float(cfg["SFQL"]["learning_rate_w"]),
        target_update_ev=1000,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_true_reward=cfg["SFQL"].getboolean("use_true_reward")
    )


    buffer = ReplayBuffer(n_samples=200000, n_batch=n_batch)
    agent = SFDQN(
            deep_sf=sf,
            buffer=buffer,
            gamma=gamma,
            T=T,
            epsilon=epsilon,
            epsilon_decay=1.0,
            epsilon_min=0.05,
            encoding=train_tasks[0].encode,   
            print_ev=2000,
            save_ev=200
        )


    agent.reset()
    agent.train(train_tasks=train_tasks, n_samples=n_samples, test_tasks=[])
    
    return (agent,train_tasks,test_task)