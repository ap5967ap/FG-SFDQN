import os
import torch
import ast
import json
import configparser
import sys
sys.path.append('..')
from features.deep import DeepSF
from features.deep_fg import DeepFGSF
from tasks.reacher import Reacher
from agents.buffer import ReplayBuffer, ConditionalReplayBuffer
from agents.sfdqn import SFDQN
from agents.fgsfdqn import FGSFDQN
from utils.config import *


reacher_config()
config_params = load_config()
cfg = config_params
gen_params = config_params['GENERAL']
n_samples = int(gen_params['n_samples'])
task_params = config_params['TASK']
goals = ast.literal_eval(task_params['train_targets'])
test_goals = ast.literal_eval(task_params['test_targets'])
all_goals = goals + test_goals
agent_params = config_params['AGENT']
dqn_params = config_params['QL']
sfdqn_params = config_params['SFQL']
fgsfdqn_params = config_params['FGSF']

def generate_tasks(include_target):
    train_tasks = [Reacher(all_goals, i, include_target) for i in range(len(goals))]
    test_tasks = [Reacher(all_goals, i + len(goals), include_target) for i in range(len(test_goals))]
    return train_tasks, test_tasks

train_tasks, test_tasks = generate_tasks(True)

def make_agent(name, algo_type, cfg, input_dim, n_actions, n_features, encoding_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer_size = int(cfg["GENERAL"]["buffer_size"])
    n_batch = int(cfg["GENERAL"]["n_batch"])
    gamma = float(cfg["AGENT"]["gamma"])
    epsilon = float(cfg["AGENT"]["epsilon"])
    T = int(cfg["AGENT"]["T"])
    
    if name == "SFDQN":
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


cfg = load_config()
tasks = train_tasks
n_samples = int(cfg["GENERAL"]["n_samples"])

input_dim = tasks[0].encode_dim()
n_actions = tasks[0].action_count()
n_features = tasks[0].feature_dim()

final_scores = {}


sequential_agents = [("SFDQN", None),("FG-SFDQN", "alg1")]

def evaluate_agent(agent, tasks, n_episodes=10):
    results = []
    for i, task in enumerate(tasks):
        total_r = 0.0
        for _ in range(n_episodes):
            # print(agent,task)
            total_r += agent.test_agent(task, max_steps=150)
        avg_r = total_r / n_episodes
        results.append(avg_r)
    return results

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


with open("final_scores_reacher.json", "w") as f:
    json.dump(final_scores, f, indent=4)



