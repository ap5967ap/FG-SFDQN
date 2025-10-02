import configparser
import ast
import os
import time
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers, losses
from keras.models import load_model
from tasks.gridworld import Shapes
from agents.buffer import ReplayBuffer
from agents.dqn import DQN


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





def main():
    def mlp_build():
        model = models.Sequential([
            layers.Input(shape=(task.encode_dim(),)),layers.Dense(128, activation="relu"),layers.Dense(128, activation="relu"),
            layers.Dense(task.action_count(), activation="linear")
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),loss='mse')
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

