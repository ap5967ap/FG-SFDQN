import configparser
import ast
import numpy as np
import torch
import matplotlib.pyplot as plt

from tasks.gridworld import Shapes
from agents.buffer import ReplayBuffer, ConditionalReplayBuffer
from agents.fgsfdqn import FGSFDQN
from features.deep_fg import DeepFGSF 

def load_cfg(path="configs/config.cfg"):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    conf = {}
    conf["n_samples"] = cfg.getint("GENERAL", "n_samples")
    conf["n_batch"] = cfg.getint("GENERAL", "n_batch")
    conf["maze"] = np.array(ast.literal_eval(cfg.get("TASK", "maze")), dtype=str)
    conf["gamma"] = cfg.getfloat("AGENT", "gamma")
    conf["epsilon"] = cfg.getfloat("AGENT", "epsilon")
    conf["T"] = cfg.getint("AGENT", "T")
    conf["sf_lr"] = cfg.getfloat("SFQL", "learning_rate")
    conf["n_averaging"] = cfg.getint("FGSF", "n_averaging", fallback=1)
    return conf

def run_experiment(algo_mode="alg1", n_avg=1):
    print(f"\n--- Running Experiment: {algo_mode} (N={n_avg}) ---")
    conf = load_cfg("configs/config.cfg")
    maze = conf["maze"]
    
    # Define Tasks
    task1 = Shapes(maze=maze, shape_rewards={'1': 1.0, '2': 0.0, '3': 0.0})
    task2 = Shapes(maze=maze, shape_rewards={'1': 0.0, '2': 1.0, '3': 0.0})
    train_tasks = [task1, task2]
    
    input_dim = task1.encode_dim()
    output_dim = task1.action_count()
    
    buffer = ConditionalReplayBuffer(n_samples=conf['n_samples'], n_batch=conf['n_batch'])
    sf_engine = DeepFGSF(input_dim, output_dim, learning_rate=conf["sf_lr"], n_averaging=n_avg)
    
    agent = FGSFDQN(sf_engine, buffer, 
                    gamma=conf["gamma"], 
                    epsilon=conf["epsilon"], 
                    T=conf["T"], 
                    encoding=task1.encode,
                    algorithm=algo_mode,
                    n_averaging=n_avg)
    
    agent.reset()
    
    history = []
    def test_callback(step):
        """Helper to evaluate agent on both tasks."""
        if step % 1000 == 0:
            rs = [agent.test_agent(t) for t in train_tasks]
            avg_r = np.mean(rs)
            history.append(avg_r)
            print(f"Step {step}: Avg Reward {avg_r:.4f}")

    steps_per_task = 15000 
    
    if algo_mode == 'alg1':
        # Sequential Training
        
        print("Training Task 1...")
        agent.reset()
        agent.add_training_task(task1)
        agent.set_active_training_task(0)
        
        for t in range(steps_per_task):
            agent.next_sample()
            test_callback(t)
            
        print("Training Task 2...")
        agent.add_training_task(task2)
        agent.set_active_training_task(1)
        
        for t in range(steps_per_task):
            agent.next_sample()
            test_callback(t + steps_per_task)
            
    else:
        # Randomized Training
        total_steps = steps_per_task * len(train_tasks)
        
        agent.reset()
        for t in train_tasks: agent.add_training_task(t)
        
        # Pre-fill
        agent.set_active_training_task(0)
        for _ in range(64): agent.next_sample()

        for t in range(total_steps):
            # Sample Task & Step
            i = np.random.randint(len(train_tasks))
            agent.set_active_training_task(i)
            agent.next_sample() 
            
            # Update
            if algo_mode == 'alg2':
                batch = buffer.replay()
                if batch:
                    agent._update_batch_grouped_by_prior(batch, i)
                    
            elif algo_mode == 'alg3':
                pivot = buffer.sample_pivot()
                if pivot:
                    p_s, p_a, _, _, _ = pivot
                    cond_batch = buffer.sample_conditional(p_s, p_a, n_avg)
                    
                    if cond_batch:
                        _, _, _, c_next_states, _ = cond_batch
                        mean_next_s = np.mean(c_next_states, axis=0)
                        
                        c = agent._get_gpi_policy(mean_next_s, i) 
                        c = c[0]
                        
                        task_c = c if c != i else None
                        agent.sf.update_averaged(p_s, p_a, cond_batch, i, task_c)
            
            test_callback(t)

    return history

if __name__ == "__main__":
    # Alg 1: Sequential
    h1 = run_experiment("alg1")
    
    # Alg 2: Randomized, Mini-batch Independent
    h2 = run_experiment("alg2")
    
    # Alg 3: Randomized, Averaged Target (N=5)
    h3 = run_experiment("alg3", n_avg=5)
    
    plt.figure(figsize=(10,6))
    plt.plot(h1, label='Alg 1 (Sequential)')
    plt.plot(h2, label='Alg 2 (Rand, Indep)')
    plt.plot(h3, label='Alg 3 (Rand, Avg N=5)')
    
    plt.axvline(x=15, color='gray', linestyle=':', label='Task Switch (Seq)')
    plt.xlabel("Steps (x1000)")
    plt.ylabel("Avg Reward")
    plt.legend()
    plt.title("Faithful Full-Gradient SF Comparison")
    plt.show()