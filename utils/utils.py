import os
import torch
import numpy as np

def save_agent_weights(agent, agent_name, trial_id, root="weights"):
    """
    Saves weights for DQN, SFDQN, or FGSFDQN agents.
    Specific support for DeepSF/DeepFGSF 'psi' structure.
    """
    path = os.path.join(root, agent_name)
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"trial_{trial_id}.pt")
    
    payload = {
        "agent_name": agent_name,
        "trial_id": trial_id,
        "algorithm": getattr(agent, "algorithm", "standard")
    }

    # --- 1. DQN Handling ---
    q_net = getattr(agent, "q", getattr(agent, "Q", None))
    if q_net is not None and isinstance(q_net, torch.nn.Module):
        payload["q_network"] = q_net.state_dict()
        
    # --- 2. SF / FGSF Handling ---
    if hasattr(agent, "sf"):
        sf_module = agent.sf
        
        # Priority 1: Save 'fit_w' (List of weights for all tasks)
        if hasattr(sf_module, "fit_w"):
            payload["reward_weights_fit"] = sf_module.fit_w 
        # Priority 2: Fallback to single 'w'
        elif hasattr(sf_module, "w"):
            w = sf_module.w
            if torch.is_tensor(w):
                payload["reward_weights"] = w.detach().cpu().numpy()
            else:
                payload["reward_weights"] = w

        # Save PSI (Successor Feature Networks)
        if hasattr(sf_module, "psi"):
            payload["psi_networks"] = {}
            for i, (model, target_model, optimizer) in enumerate(sf_module.psi):
                payload["psi_networks"][i] = {
                    "model": model.state_dict(),
                    "target": target_model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }

    torch.save(payload, filename)
    print(f"Saved: {filename}")
    
    
def load_agent_weights(agent, filename):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    checkpoint = torch.load(filename)
    print(f"Loading {checkpoint.get('agent_name', 'Agent')} (Trial {checkpoint.get('trial_id', '?')})...")

    # --- Load SF / FGSF ---
    if hasattr(agent, "sf") and "psi_networks" in checkpoint:
        sf_module = agent.sf
        psi_data = checkpoint["psi_networks"]
        
        # Strict Check: Agent must have tasks initialized before loading
        assert len(sf_module.psi) == len(psi_data), (
            f"Shape Mismatch: Agent has {len(sf_module.psi)} policies initialized, "
            f"but checkpoint contains {len(psi_data)}. "
            "Initialize tasks (agent.add_task) before loading."
        )

        for i, data in psi_data.items():
            idx = int(i) # Safety cast
            if idx < len(sf_module.psi):
                model, target, opt = sf_module.psi[idx]
                model.load_state_dict(data["model"])
                target.load_state_dict(data["target"])
                opt.load_state_dict(data["optimizer"])
    
    # --- Load Rewards ---
    if hasattr(agent, "sf"):
        if "reward_weights_fit" in checkpoint:
            # Restore full list of task weights
            agent.sf.fit_w = checkpoint["reward_weights_fit"]
        elif "reward_weights" in checkpoint:
            # Fallback for single task weight
            w_data = checkpoint["reward_weights"]
            if hasattr(agent.sf, "w") and torch.is_tensor(agent.sf.w):
                agent.sf.w.data = torch.from_numpy(w_data).to(agent.sf.w.device)
            else:
                agent.sf.w = w_data

    # --- Load DQN ---
    if hasattr(agent, "q") and "q_network" in checkpoint:
        agent.q.load_state_dict(checkpoint["q_network"])

    print("Weights loaded successfully.")