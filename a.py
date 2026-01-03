
import sys
sys.path.append(rf'/kaggle/input/advrlclone/adv_rl (Copy)')


import os
import ast
import configparser
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

# --- Import Agents and Tasks ---
# Assumes the file structure provided in the prompt exists
from tasks.gridworld import Shapes
from features.deep import DeepSF
from features.deep_fg import DeepFGSF
from agents.buffer import ReplayBuffer, ConditionalReplayBuffer
from agents.dqn import DQN
from agents.sfdqn import SFDQN
from agents.fgsfdqn import FGSFDQN
from agents.dqn import DQN


CONFIG_CONTENT = """
[GENERAL]
n_samples=10000
n_tasks=6
n_trials=5
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
    """Generates a sequence of tasks (MDPs)."""
    maze = np.array(ast.literal_eval(cfg["TASK"]["maze"]), dtype=str)
    
    # Diverse reward functions
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
    # Standard: Calculate Phi and update Reward model
        phi = self.phi(s, a, s1)
        if isinstance(phi, torch.Tensor):
            phi = phi.detach().cpu().numpy()
        self.sf.update_reward(phi, r, self.task_index)
    
        # Standard: Store in buffer
        self.buffer.append(s_enc, a, phi, s1_enc, gamma)
    
        # If not Alg1 or Alg1_averaged, do nothing here (training happens elsewhere)
        if self.algorithm not in ['alg1', 'alg1_averaged']:
            return
    
        # --- AVERAGED UPDATE LOGIC (New for Alg 1) ---
        if self.n_averaging > 1:
            # Robustly shape pivot state to (1, dim)
            try:
                p_s = np.asarray(s_enc).reshape(1, -1)
            except Exception:
                # fallback: try wrapping into 1D then reshape
                p_s = np.array([s_enc]).reshape(1, -1)
    
            # Defensive: ensure buffer supports conditional sampling
            if not hasattr(self.buffer, "sample_conditional"):
                # fallback to standard replay update
                batch = self.buffer.replay()
                if batch:
                    self._update_batch_grouped_by_prior(batch, self.task_index)
                return
    
            # Query the buffer for N samples of (s, a)
            cond_batch = self.buffer.sample_conditional(p_s, a, self.n_averaging)
    
            if cond_batch:
                # Expected cond_batch: (p_s, p_a, c_phis, c_next_states, c_gammas)
                _, _, _, c_next_states, _ = cond_batch
    
                # Ensure numpy array and shape (N, dim)
                c_next_states = np.asarray(c_next_states)
                if c_next_states.ndim == 3 and c_next_states.shape[1] == 1:
                    # squeeze singleton middle dim -> (N, dim)
                    c_next_states = c_next_states.squeeze(1)
                elif c_next_states.ndim == 1:
                    # single sample -> shape (1, dim)
                    c_next_states = c_next_states.reshape(1, -1)
    
                # Compute mean over N (result is 1D array of length enc_dim)
                mean_next_state_feature = np.mean(c_next_states, axis=0)
    
                # Select GPI prior 'c' based on mean feature (GPI expects either 1D or 2D)
                c = self._get_gpi_policy(mean_next_state_feature, self.task_index)[0]
    
                # Determine tasks to update (None means only current task)
                task_c = c if c != self.task_index else None
    
                # Perform averaged full-gradient update (DeepFGSF expects pivot as shape (1,dim))
                self.sf.update_averaged(p_s, a, cond_batch, self.task_index, task_c)
                # Early return to avoid regular replay update
                return
    
        # --- FALLBACK / STANDARD LOGIC ---
        batch = self.buffer.replay()
        if batch:
            self._update_batch_grouped_by_prior(batch, self.task_index)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate=1e-3, device=None):
        super(MLP, self).__init__()
        if device is None:
             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
             self.device = device
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
        
        # --- Task & Config Setup ---
        input_dim = tasks[0].encode_dim()
        n_actions = tasks[0].action_count()
        n_features = tasks[0].feature_dim()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        buffer_size = int(cfg["GENERAL"]["buffer_size"])
        n_batch = int(cfg["GENERAL"]["n_batch"])
        gamma = float(cfg["AGENT"]["gamma"])
        epsilon = float(cfg["AGENT"]["epsilon"])
        T = int(cfg["AGENT"]["T"])
        
        # Extract params
        algo = agent_params.get("algorithm", "alg1")
        if "n_averaging" in override_params:
            n_avg = override_params["n_averaging"]
        else:
            n_avg = int(cfg["FGSF"]["n_averaging"])

        agent = None

        # --- AGENT SELECTION LOGIC ---
        if agent_name == "DQN":
            def model_builder():
                return MLP(input_dim, n_actions, learning_rate=float(cfg["QL"]["learning_rate"]), device=device)

            buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
            agent = DQN(
                model_builder,
                buffer,
                gamma=gamma,
                epsilon=epsilon,
                T=T,
                encoding=tasks[0].encode,
                save_ev=save_ev
            )

        elif agent_name == "SFDQN":
            # 1. Baseline SFDQN
            sf = DeepSF(
                input_dim=input_dim, n_actions=n_actions, n_features=n_features,
                learning_rate=float(cfg["SFQL"]["learning_rate"]),
                learning_rate_w=float(cfg["SFQL"]["learning_rate_w"]),
                device=device, use_true_reward=cfg["SFQL"].getboolean("use_true_reward")
            )
            buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
            agent = SFDQN(sf, buffer, gamma=gamma, epsilon=epsilon, T=T, encoding=tasks[0].encode, save_ev=save_ev)
            
        elif agent_name == "AvgFGSFDQN":
            # 2. Your New Averaged Class (Alg 1 Averaged)
            sf = DeepFGSF(
                input_dim=input_dim, n_actions=n_actions, n_features=n_features,
                learning_rate=float(cfg["SFQL"]["learning_rate"]),
                learning_rate_prior=float(cfg["SFQL"].get("learning_rate_prior", cfg["SFQL"]["learning_rate"])),
                learning_rate_w=float(cfg["SFQL"]["learning_rate_w"]),
                device=device, use_true_reward=cfg["SFQL"].getboolean("use_true_reward")
            )
            # MUST use Conditional Buffer for averaging
            buffer = ConditionalReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
            
            agent = AvgFGSFDQN(sf, buffer, gamma=gamma, T=T, epsilon=epsilon, encoding="task",
                               algorithm="alg1_averaged", n_averaging=n_avg, save_ev=save_ev)

        else:
            # 3. Standard FGSFDQN (Alg 1, 2, 3)
            sf = DeepFGSF(
                input_dim=input_dim, n_actions=n_actions, n_features=n_features,
                learning_rate=float(cfg["SFQL"]["learning_rate"]),
                learning_rate_prior=float(cfg["SFQL"].get("learning_rate_prior", cfg["SFQL"]["learning_rate"])),
                learning_rate_w=float(cfg["SFQL"]["learning_rate_w"]),
                device=device, use_true_reward=cfg["SFQL"].getboolean("use_true_reward")
            )
            
            # Use Conditional Buffer only if Algo 3 is requested explicitly
            if algo in ["alg3"]:
                buffer = ConditionalReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
            else:
                buffer = ReplayBuffer(n_samples=buffer_size, n_batch=n_batch)
                
            agent = FGSFDQN(sf, buffer, gamma=gamma, T=T, epsilon=epsilon, encoding="task",
                            algorithm=algo, n_averaging=n_avg, save_ev=save_ev)

        agent.reset()
        trial_history = []
        
        # --- Training Loop ---
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

        save_agent_weights(agent, agent_name, trial + 1)
        all_trials_data.append(trial_history)
        
    return np.array(all_trials_data)


cfg = load_config()
tasks = build_task_sequence(cfg)

# n_trials = int(cfg["GENERAL"]["n_trials"])
results = {}



def plot_ablation(results, cfg, tasks, averaging_values, with_std=True, filename="plot.png"):
    plt.figure(figsize=(12, 7))

    # X-axis
    n_samples = int(cfg["GENERAL"]["n_samples"])
    total_steps = n_samples * len(tasks)
    first_key = next(iter(results))
    x_axis = np.linspace(0, total_steps, len(results[first_key][0]))

    # --- Baselines ---
    sf_mean, sf_std = results["SFDQN"]
    plt.plot(x_axis, sf_mean, label='SFDQN', color='black', linestyle='--', linewidth=2)
    if with_std:
        plt.fill_between(
            x_axis, sf_mean - sf_std, sf_mean + sf_std,
            color='black', alpha=0.1
        )

    fg_mean, fg_std = results["FGSFDQN (Alg1)"]
    plt.plot(
        x_axis, fg_mean,
        label='FGSF Alg1 (No Avg)',
        color='gray', linestyle='-.', linewidth=2
    )
    if with_std:
        plt.fill_between(
            x_axis, fg_mean - fg_std, fg_mean + fg_std,
            color='gray', alpha=0.1
        )

    # --- Averaging Experiments ---
    colors = plt.cm.viridis(np.linspace(0.2, 1, len(averaging_values)))
    for i, n in enumerate(averaging_values):
        label = f"AvgFGSFDQN (N={n})"
        mean, std = results[label]

        plt.plot(
            x_axis, mean,
            label=f'Alg1 Avg (N={n})',
            color=colors[i],
            linewidth=2
        )
        if with_std:
            plt.fill_between(
                x_axis, mean - std, mean + std,
                color=colors[i], alpha=0.1
            )

    title_suffix = "With Std Dev" if with_std else "Mean Only"
    plt.title(f"Ablation: Effect of Averaging on Algorithm 1 ({title_suffix})")
    plt.xlabel("Training Steps")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()



def run_all_experiments(cfg, tasks, n_trials=5):
    """
    Runs all agents exactly once and returns raw trial data.
    """
    raw = {}

    print("\n============================================")
    print("RUNNING ALL EXPERIMENTS (ONCE)")
    print("============================================")

    # --- DQN ---
    print("\nRunning DQN...")
    raw["DQN"] = run_experiment("DQN", {}, cfg, tasks, n_trials)

    # --- SFDQN ---
    print("\nRunning SFDQN...")
    raw["SFDQN"] = run_experiment("SFDQN", {}, cfg, tasks, n_trials)

    # --- FGSFDQN Alg1 (NO averaging) ---
    print("\nRunning FGSFDQN (Alg1)...")
    raw["FGSFDQN"] = run_experiment(
        "FGSFDQN",
        {"algorithm": "alg1"},
        cfg,
        tasks,
        n_trials,
        override_params={"n_averaging": 1}
    )

    # --- Averaged Alg1 variants ---
    for n in [5, 10, 20]:
        print(f"\nRunning AvgFGSFDQN (N={n})...")
        raw[f"AvgFGSFDQN (N={n})"] = run_experiment(
            "AvgFGSFDQN",
            {"algorithm": "alg1_averaged"},
            cfg,
            tasks,
            n_trials,
            override_params={"n_averaging": n}
        )

    return raw

def save_raw_results(raw_results, filename="raw_results.npz"):
    np.savez_compressed(filename, **raw_results)
    print(f"[OK] Raw results saved to {filename}")

import os
import torch

def save_agent_weights(agent, agent_name, trial_id, root="weights"):
    """
    Saves model weights for one agent and one trial.
    """
    path = os.path.join(root, agent_name)
    os.makedirs(path, exist_ok=True)

    filename = os.path.join(path, f"trial_{trial_id}.pt")

    payload = {}

    # --- DQN ---
    if hasattr(agent, "q"):
        payload["q_network"] = agent.q.state_dict()

    # --- SF-based agents ---
    if hasattr(agent, "sf"):
        payload["sf_network"] = agent.sf.phi_net.state_dict()
        payload["reward_weights"] = agent.sf.w.copy()

        if hasattr(agent.sf, "prior_nets"):
            payload["prior_nets"] = {
                i: net.state_dict()
                for i, net in enumerate(agent.sf.prior_nets)
            }

    torch.save(payload, filename)
    print(f"[OK] Saved weights → {filename}")
