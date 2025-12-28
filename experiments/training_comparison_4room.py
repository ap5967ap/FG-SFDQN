import ast
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import seaborn as sns
from tasks.gridworld import Shapes
from features.deep import DeepSF
from features.deep_fg import DeepFGSF
from agents.buffer import ReplayBuffer, ConditionalReplayBuffer
from agents.dqn import DQN
from agents.sfdqn import SFDQN
from agents.fgsfdqn import FGSFDQN
from utils.config import *

fourroom_config()
config_params = load_config()
cfg = config_params

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
    
gen_params = config_params['GENERAL']
n_samples = int(gen_params['n_samples'])
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


def run_experiment(agent_name, agent_params, cfg, tasks, n_trials):
    print(f"--- Running {agent_name} ---")
    n_samples = int(cfg["GENERAL"]["n_samples"])
    save_ev = int(cfg["AGENT"]["save_ev"])
    total_data_points = (n_samples * len(tasks)) // save_ev
    all_trials_data = []
    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials}")
        
        input_dim = tasks[0].encode_dim()
        n_actions = tasks[0].action_count()
        n_features = tasks[0].feature_dim()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        agent = None
        buffer_size = int(cfg["GENERAL"]["buffer_size"])
        n_batch = int(cfg["GENERAL"]["n_batch"])
        gamma = float(cfg["AGENT"]["gamma"])
        epsilon = float(cfg["AGENT"]["epsilon"])
        T = int(cfg["AGENT"]["T"])
        if agent_name == "DQN":
            def model_builder():
                return MLP(input_dim, n_actions, learning_rate=float(cfg["QL"]["learning_rate"]))
            
            buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
            agent = DQN(model_builder, buffer, gamma=gamma, epsilon=epsilon, T=T, encoding=tasks[0].encode, save_ev=save_ev)
        elif agent_name == "SFDQN":
            sf = DeepSF(
                input_dim=input_dim, n_actions=n_actions, n_features=n_features,
                learning_rate=float(cfg["SFQL"]["learning_rate"]),
                learning_rate_w=float(cfg["SFQL"]["learning_rate_w"]),
                device=device, use_true_reward=cfg["SFQL"].getboolean("use_true_reward")
            )
            buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
            agent = SFDQN(sf, buffer, gamma=gamma, epsilon=epsilon, T=T, encoding=tasks[0].encode, save_ev=save_ev)
        else:
            algo = agent_params.get("algorithm")
            n_avg = int(cfg["FGSF"]["n_averaging"])
            
            sf = DeepFGSF(
                input_dim=input_dim, n_actions=n_actions, n_features=n_features,
                learning_rate=float(cfg["SFQL"]["learning_rate"]),
                learning_rate_prior=float(cfg["SFQL"].get("learning_rate_prior", cfg["SFQL"]["learning_rate"])),
                learning_rate_w=float(cfg["SFQL"]["learning_rate_w"]),
                device=device, use_true_reward=cfg["SFQL"].getboolean("use_true_reward")
            )
            
            if algo in ["alg3", "alg4"]:
                buffer = ConditionalReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
            else:
                buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
            agent = FGSFDQN(sf, buffer, gamma=gamma, T=T, epsilon=epsilon, encoding="task",
                            algorithm=algo, n_averaging=n_avg, save_ev=save_ev)
        agent.reset()
        trial_history = []
        
        for t_idx, task in enumerate(tasks):
            agent.cum_reward = 0 
            agent.train_on_task(task, n_samples=n_samples)
            points_per_task = n_samples // save_ev
            curr_hist = agent.cum_reward_hist
            if len(curr_hist) >= points_per_task:
                 task_data = curr_hist[-points_per_task:]
            else:
                 task_data = curr_hist
            trial_history.extend(task_data)

        if len(trial_history) != total_data_points:
            if len(trial_history) < total_data_points:
                 trial_history.extend([trial_history[-1]] * (total_data_points - len(trial_history)))
            else:
                 trial_history = trial_history[:total_data_points]
        all_trials_data.append(trial_history)
    return np.array(all_trials_data)


n_trials = 1
def main():
    results = {}
    results["DQN"] = run_experiment("DQN", {}, cfg, tasks, n_trials)
    results["FG-SFDQN (Alg 1)"] = run_experiment("FG-SFDQN", {"algorithm": "alg1"}, cfg, tasks, n_trials)
    results["SFDQN"] = run_experiment("SFDQN", {}, cfg, tasks, n_trials)
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 7))

    save_ev = int(cfg["AGENT"]["save_ev"])
    total_samples = int(cfg["GENERAL"]["n_samples"]) * int(cfg["GENERAL"]["n_tasks"])

    x_axis = np.linspace(0, total_samples, num=results["SFDQN"].shape[1])
    styles = {
        "DQN": {"color": "#ff7f0e", "label": "DQN"},
        "SFDQN": {"color": "#7f7f7f", "label": "SFDQN"},
        "FG-SFDQN (Alg 1)": {"color": "#1f77b4", "label": "FG-SFDQN (Alg 1)"},
    }    
    
    for agent_name, style in styles.items():
        if agent_name not in results: continue
        
        data = results[agent_name]
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        plt.plot(x_axis, mean, label=style["label"], color=style["color"], linewidth=2)
        plt.fill_between(x_axis, mean - std, mean + std, color=style["color"], alpha=0.15)

    plt.title("Cumulative Training Reward Per Task", fontsize=18)
    plt.xlabel("Training Samples", fontsize=14)
    plt.ylabel("Cumulative Reward", fontsize=14)
    plt.legend(frameon=True, loc='upper left', fontsize=12)

    samples_per_task = int(cfg["GENERAL"]["n_samples"])
    n_tasks = int(cfg["GENERAL"]["n_tasks"])
    for i in range(1, n_tasks):
        plt.axvline(x=i * samples_per_task, color='black', linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig("cumulative_reward_comparison_4room.png", dpi=300)
    plt.show()
