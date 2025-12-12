import os
import ast
import configparser
import numpy as np
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

CONFIG_CONTENT = """
[GENERAL]
n_samples=10000
n_tasks=6
n_trials=1
n_batch=512
buffer_size=200000

[TASK]
# Standard Barreto et al. Gridworld layout
maze=[
    ['1', ' ', ' ', ' ', ' ', '2', 'X', ' ', ' ', ' ', ' ', ' ', 'G'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', '1', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['2', ' ', ' ', ' ', ' ', '3', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['X', 'X', '3', ' ', 'X', 'X', 'X', 'X', 'X', ' ', '1', 'X', 'X'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', '2', ' ', ' ', ' ', ' ', '3'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', '2', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['_', ' ', ' ', ' ', ' ', ' ', 'X', '3', ' ', ' ', ' ', ' ', '1']]

[AGENT]
gamma=0.95
epsilon=0.55
T=200
print_ev=2000
save_ev=200

[SFQL]
learning_rate=0.001
learning_rate_prior=0.00001
learning_rate_w=0.5
use_true_reward=False
hidden_units=128

[QL]
learning_rate=0.5

[FGSF]
n_averaging=5
"""

if not os.path.exists("configs"):
    os.makedirs("configs")
with open("configs/config.cfg", "w") as f:
    f.write(CONFIG_CONTENT)

def load_config(path="configs/config.cfg"):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg

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

class AvgFGSFDQN(FGSFDQN):
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        phi = self.phi(s, a, s1)
        if isinstance(phi, torch.Tensor):
            phi = phi.detach().cpu().numpy()
        self.sf.update_reward(phi, r, self.task_index)
    
        self.buffer.append(s_enc, a, phi, s1_enc, gamma)
        if self.algorithm not in ['alg1', 'alg1_averaged']:
            return
        if self.n_averaging > 1:
            try:
                p_s = np.asarray(s_enc).reshape(1, -1)
            except Exception:
                p_s = np.array([s_enc]).reshape(1, -1)
            if not hasattr(self.buffer, "sample_conditional"):
                batch = self.buffer.replay()
                if batch:
                    self._update_batch_grouped_by_prior(batch, self.task_index)
                return
            cond_batch = self.buffer.sample_conditional(p_s, a, self.n_averaging)
            if cond_batch:
                _, _, _, next_c, _ = cond_batch
                next_c = np.asarray(next_c)
                if next_c.ndim == 3 and next_c.shape[1] == 1:
                    next_c = next_c.squeeze(1)
                elif next_c.ndim == 1:
                    next_c = next_c.reshape(1, -1)
                phi_dash = np.mean(next_c, axis=0)
                c = self._get_gpi_policy(phi_dash, self.task_index)[0]
                task_c = c if c != self.task_index else None
                self.sf.update_averaged(p_s, a, cond_batch, self.task_index, task_c)
                return
        batch = self.buffer.replay()
        if batch:
            self._update_batch_grouped_by_prior(batch, self.task_index)

def run_experiment(agent_name, agent_params, cfg, tasks, n_trials, override_params=None):
    if override_params is None:
        override_params = {}

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
        
        buffer_size = int(cfg["GENERAL"]["buffer_size"])
        n_batch = int(cfg["GENERAL"]["n_batch"])
        gamma = float(cfg["AGENT"]["gamma"])
        epsilon = float(cfg["AGENT"]["epsilon"])
        T = int(cfg["AGENT"]["T"])
        
        algo = agent_params.get("algorithm", "alg1")
        if "n_averaging" in override_params:
            n_avg = override_params["n_averaging"]
        else:
            n_avg = int(cfg["FGSF"]["n_averaging"])

        agent = None

        if agent_name == "SFDQN":
            sf = DeepSF(
                input_dim=input_dim, n_actions=n_actions, n_features=n_features,
                learning_rate=float(cfg["SFQL"]["learning_rate"]),
                learning_rate_w=float(cfg["SFQL"]["learning_rate_w"]),
                device=device, use_true_reward=cfg["SFQL"].getboolean("use_true_reward")
            )
            buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
            agent = SFDQN(sf, buffer, gamma=gamma, epsilon=epsilon, T=T, encoding=tasks[0].encode, save_ev=save_ev)
            
        elif agent_name == "AvgFGSFDQN":
            sf = DeepFGSF(
                input_dim=input_dim, n_actions=n_actions, n_features=n_features,
                learning_rate=float(cfg["SFQL"]["learning_rate"]),
                learning_rate_prior=float(cfg["SFQL"].get("learning_rate_prior", cfg["SFQL"]["learning_rate"])),
                learning_rate_w=float(cfg["SFQL"]["learning_rate_w"]),
                device=device, use_true_reward=cfg["SFQL"].getboolean("use_true_reward")
            )
            buffer = ConditionalReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
            
            agent = AvgFGSFDQN(sf, buffer, gamma=gamma, T=T, epsilon=epsilon, encoding="task",
                               algorithm="alg1_averaged", n_averaging=n_avg, save_ev=save_ev)

        else:
            sf = DeepFGSF(
                input_dim=input_dim, n_actions=n_actions, n_features=n_features,
                learning_rate=float(cfg["SFQL"]["learning_rate"]),
                learning_rate_prior=float(cfg["SFQL"].get("learning_rate_prior", cfg["SFQL"]["learning_rate"])),
                learning_rate_w=float(cfg["SFQL"]["learning_rate_w"]),
                device=device, use_true_reward=cfg["SFQL"].getboolean("use_true_reward")
            )
            
            if algo in ["alg3"]:
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

cfg = load_config()
tasks = build_task_sequence(cfg)

n_trials = int(cfg["GENERAL"]["n_trials"])
results = {}


def run_alg1_ablation(cfg, tasks, n_trials=3):
    results = {}
    print("\n============================================")
    print("STARTING ABLATION: Algorithm 1 Averaging Study")
    print("============================================")

    print("\nRunning Baseline SFDQN...")
    sfdqn_data = run_experiment("SFDQN", {}, cfg, tasks, n_trials)
    results["SFDQN"] = (np.mean(sfdqn_data, axis=0), np.std(sfdqn_data, axis=0))

    print("\nRunning FGSFDQN (Alg 1 - No Averaging)...")
    fg_alg1_data = run_experiment("FGSFDQN", {"algorithm": "alg1"}, cfg, tasks, n_trials, override_params={"n_averaging": 1})
    results["FGSFDQN (Alg1)"] = (np.mean(fg_alg1_data, axis=0), np.std(fg_alg1_data, axis=0))

    averaging_values = [5, 10, 20]
    for n in averaging_values:
        print(f"\nRunning AvgFGSFDQN (Alg 1 Averaged) with N={n}...")
        
        data = run_experiment(
            agent_name="AvgFGSFDQN",
            agent_params={"algorithm": "alg1_averaged"},
            cfg=cfg,
            tasks=tasks,
            n_trials=n_trials,
            override_params={"n_averaging": n}
        )
        results[f"AvgFGSFDQN (N={n})"] = (np.mean(data, axis=0), np.std(data, axis=0))

    plt.figure(figsize=(12, 7))
    
    n_samples = int(cfg["GENERAL"]["n_samples"])
    total_steps = n_samples * len(tasks)
    first_key = next(iter(results))
    x_axis = np.linspace(0, total_steps, len(results[first_key][0]))

    sf_mean, sf_std = results["SFDQN"]
    plt.plot(x_axis, sf_mean, label='SFDQN', color='black', linestyle='--', linewidth=2)
    plt.fill_between(x_axis, sf_mean - sf_std, sf_mean + sf_std, color='black', alpha=0.1)

    fg_mean, fg_std = results["FGSFDQN (Alg1)"]
    plt.plot(x_axis, fg_mean, label='FGSF Alg1 (No Avg)', color='gray', linestyle='-.', linewidth=2)
    plt.fill_between(x_axis, fg_mean - fg_std, fg_mean + fg_std, color='gray', alpha=0.1)

    colors = plt.cm.viridis(np.linspace(0.2, 1, len(averaging_values)))
    for i, n in enumerate(averaging_values):
        label = f"AvgFGSFDQN (N={n})"
        mean, std = results[label]
        plt.plot(x_axis, mean, label=f'Alg1 Avg (N={n})', color=colors[i], linewidth=2)
        plt.fill_between(x_axis, mean - std, mean + std, color=colors[i], alpha=0.1)

    plt.title("Ablation: Effect of Averaging on Algorithm 1 (Sequential)")
    plt.xlabel("Training Steps")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    filename = "ablation_alg1_averaging.png"
    plt.savefig(filename)
    print(f"\nAblation complete. Plot saved to {filename}")
    plt.show()