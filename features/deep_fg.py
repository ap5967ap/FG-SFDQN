import torch
import torch.nn as nn
import numpy as np
from features.deep import DeepSF

class DeepFGSF(DeepSF):
    def __init__(self, learning_rate_prior=None, *args, **kwargs):
        super(DeepFGSF, self).__init__(*args, **kwargs)
        self.learning_rate_prior = learning_rate_prior if learning_rate_prior is not None else self.learning_rate

    def _set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
    def _to_tensor(self, x, dtype=torch.float):
        if not isinstance(x, torch.Tensor):
            return torch.from_numpy(x).to(dtype=dtype, device=self.device)
        return x.to(dtype=dtype, device=self.device)

    def _get_next_actions_gpi(self, next_states, policy_index):
        """
        Selects actions using GPI based on the current task's reward weights
        but evaluating over all policies.
        """
        if isinstance(next_states, torch.Tensor):
            ns_np = next_states.detach().cpu().numpy()
        else:
            ns_np = next_states
            
        # Get action that maximizes Q for the current task
        q_gpi, _ = self.GPI(ns_np, policy_index)   # [Batch, n_tasks, n_actions]
        next_actions = np.argmax(np.max(q_gpi, axis=1), axis=-1)
        return torch.from_numpy(next_actions).long().to(self.device)

    def _get_next_action_prior(self, next_states, policy_index):
        """
        Selects actions using the prior policy for a specific task k:
        a' = argmax_a (psi_k(s', a) * w_k)
        """
        model, _, _ = self.psi[policy_index]
        model.eval()
        
        # Get Features for policy k
        with torch.no_grad():
            psi = model(next_states) # [Batch, Actions, Features]
            
        # Get Reward Weights for policy k
        w = torch.from_numpy(self.fit_w[policy_index]).float().to(self.device) # [Features, 1]
        
        # Compute Q = psi * w
        q_values = torch.matmul(psi, w).squeeze(-1) # [B, A, F] x [F, 1] -> [B, A, 1]
        
        # Argmax
        next_actions = torch.argmax(q_values, dim=1)
        return next_actions

    def update_single_sample(self, transition, task_i, task_c):
        """
        Updates theta^i and theta^c using a single transition (s, a, s')
        """
        states, actions, phis, next_states, gammas = transition
        
        states = self._to_tensor(states)
        actions = self._to_tensor(actions, torch.long)
        phis = self._to_tensor(phis)
        next_states = self._to_tensor(next_states)
        gammas = self._to_tensor(gammas).view(-1, 1)
        
        indices = torch.arange(states.shape[0])
        
        # Identify tasks to update
        tasks_to_update = {task_i}
        if task_c is not None:
            tasks_to_update.add(task_c)
            
        total_loss = 0
        
        for k in tasks_to_update:
            model, _, optimizer = self.psi[k]
            
            # Update LR
            lr = self.learning_rate if k == task_i else self.learning_rate_prior
            self._set_lr(optimizer, lr)

            model.train()
            optimizer.zero_grad()

            if k==task_i:
                # Current task: Use GPI action
                next_acts = self._get_next_actions_gpi(next_states, task_i)
            else:
                # Prior Task
                next_acts = self._get_next_action_prior(next_states, k)
            
            # Prediction: xi_k(s, a)
            pred_all = model(states)
            xi_s = pred_all[indices, actions, :]
            
            # Target: phi + gamma * xi_k(s', a')
            # Full Gradient: Gradients flow through xi_k(s')
            pred_next_all = model(next_states)
            xi_next = pred_next_all[indices, next_acts, :]
            
            target = phis + gammas * xi_next
            
            diff = target - xi_s
            loss = 0.5 * (diff.pow(2)).mean()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    def update_averaged(self, pivot_state, pivot_action, conditional_batch, task_i, task_c):
        """
        Implements averaging for bellman update over N samples.
        
        pivot_state: (1, dim)
        pivot_action: int
        conditional_batch: tuple of (N, dim) tensors containing N samples for (s, a)
        """
        _, _, c_phis, c_next_states, c_gammas = conditional_batch
        
        c_phis = self._to_tensor(c_phis)
        c_nexts = self._to_tensor(c_next_states)
        c_gammas = self._to_tensor(c_gammas).view(-1, 1)
        
        pivot_state_t = self._to_tensor(pivot_state)
        tasks_to_update = {task_i}
        if task_c is not None:
            tasks_to_update.add(task_c)

        for k in tasks_to_update:
            model, _, optimizer = self.psi[k]
            
            # Update LR
            lr = self.learning_rate if k == task_i else self.learning_rate_prior
            self._set_lr(optimizer, lr)

            model.train()
            optimizer.zero_grad()
            
            # Prediction: xi_k(s, a) (Single value for the pivot)
            pred_pivot = model(pivot_state_t) # [1, n_actions, dim]
            xi_s = pred_pivot[0, pivot_action, :] # [dim]
            
            # Compute Averaged Target over N samples
            if k == task_i:
                next_acts = self._get_next_actions_gpi(c_nexts, task_i)
            else:
                next_acts = self._get_next_action_prior(c_nexts, k)
            
            pred_next_all = model(c_nexts) # [N, n_actions, dim]
            indices = torch.arange(c_nexts.shape[0])
            xi_next = pred_next_all[indices, next_acts, :] # [N, dim]
            
            # Element-wise target: phi_p + gamma_p * xi(s'_p)
            targets_N = c_phis + c_gammas * xi_next
            
            # Average over N
            target_bar = torch.mean(targets_N, dim=0) # [dim]
            
            diff = target_bar - xi_s
            loss = 0.5 * (diff.pow(2)).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()