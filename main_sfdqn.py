
import configparser
import ast
import numpy as np
from tasks.gridworld import Shapes
from agents.buffer import ReplayBuffer
from agents.sfdqn import SFDQN
import torch

from features.deep import DeepSF

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
	conf["sf_lr"] = cfg.getfloat("SF", "learning_rate")
	return conf


def main():
	conf = load_cfg("configs/config.cfg")
	maze = conf["maze"]
	task = Shapes(maze=maze, shape_rewards={'1':10,'2':0,'3':0})
	buffer = ReplayBuffer(n_samples=conf['n_samples'], n_batch=conf['n_batch'])

	# Build DeepSF instance. You may need to adjust input/output dims and learning rate as per your DeepSF implementation
	if DeepSF is None:
		raise ImportError("DeepSF class not found. Please implement or import DeepSF.")
	input_dim = task.encode_dim()
	output_dim = task.action_count()
	deep_sf = DeepSF(input_dim, output_dim, learning_rate=conf.get("sf_lr", 1e-3))

	agent = SFDQN(deep_sf, buffer, gamma=conf["gamma"], epsilon=conf["epsilon"], T=conf["T"], encoding=task.encode)
	agent.reset()

	train_tasks = [task]
	n_samples = conf['n_samples']
	agent.train(train_tasks=train_tasks, n_samples=n_samples, test_tasks=[])
	return (agent, train_tasks)
