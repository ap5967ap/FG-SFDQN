# -*- coding: UTF-8 -*-
import numpy as np


class SF:
    
    def __init__(self, learning_rate_w, *args, use_true_reward=False, **kwargs):
        """
        Creates a new abstract successor feature representation.
        
        Parameters
        ----------
        learning_rate_w : float
            the learning rate to use for learning the reward weights using gradient descent
        use_true_reward : boolean
            whether or not to use the true reward weights from the environment, or learn them
            using gradient descent
        """
        self.alpha_w = learning_rate_w
        self.use_true_reward = use_true_reward
        if len(args) != 0 or len(kwargs) != 0:
            print(self.__class__.__name__ + ' ignoring parameters ' + str(args) + ' and ' + str(kwargs))
        self.reset()
            
    def build_successor(self, task, source=None):
        """
        Builds a new successor feature map for the specified task. This method should not be called directly.
        Instead, add_task should be called instead.
        
        Parameters
        ----------
        task : Task
            a new MDP environment for which to learn successor features
        source : integer
            if specified and not None, the parameters of the successor features for the task at the source
            index should be copied to the new successor features, as suggested in [1]
        
        Returns
        -------
        object : the successor feature representation for the new task, which can be a torch model, 
        a lookup table (dictionary) or another learning representation
        """
        raise NotImplementedError
        
    def get_successor(self, state, policy_index):
        """
        Evaluates the successor features in given states for the specified task.
        
        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        policy_index : integer
            the index of the task whose successor features to evaluate
        
        Returns
        -------
        np.ndarray : the evaluation of the successor features, which is of shape
        [n_batch, n_actions, n_features], where
            n_batch is the number of states in the state argument
            n_actions is the number of actions of the MDP
            n_features is the number of features in the SF representation
        """
        raise NotImplementedError
    
    def get_successors(self, state):
        """
        Evaluates the successor features in given states for all tasks.
        
        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        
        Returns
        -------
        np.ndarray : the evaluation of the successor features, which is of shape
        [n_batch, n_tasks, n_actions, n_features], where
            n_batch is the number of states in the state argument
            n_tasks is the number of tasks
            n_actions is the number of actions of the MDP
            n_features is the number of features in the SF representation
        """
        raise NotImplementedError
    
    def update_successor(self, transitions, policy_index):
        """
        Updates the successor representation by training it on the given transition.
        
        Parameters
        ----------
        transitions : object
            collection of transitions
        policy_index : integer
            the index of the task whose successor features to update
        """
        raise NotImplementedError
        
    def reset(self):
        """
        Removes all trained successor feature representations from the current object, all learned rewards,
        and all task information.
        """
        self.n_tasks = 0
        self.psi = []
        self.true_w = []
        self.fit_w = []
        self.gpi_counters = []

    def add_training_task(self, task, source=None):
        """
        Adds a successor feature representation for the specified task.
        
        Parameters
        ----------
        task : Task
            a new MDP environment for which to learn successor features
        source : integer
            if specified and not None, the parameters of the successor features for the task at the source
            index should be copied to the new successor features, as suggested in [1]
        """
        
        # add successor features to the library
        self.psi.append(self.build_successor(task, source))
        self.n_tasks = len(self.psi)
        
        # build new reward function
        true_w = task.get_w()
        self.true_w.append(true_w)
        if self.use_true_reward:
            fit_w = true_w
        else:
            n_features = task.feature_dim()
            fit_w = np.random.uniform(low=-0.01, high=0.01, size=(n_features, 1))
        self.fit_w.append(fit_w)
        
        # add statistics
        for i in range(len(self.gpi_counters)):
            self.gpi_counters[i] = np.append(self.gpi_counters[i], 0)
        self.gpi_counters.append(np.zeros((self.n_tasks,), dtype=int))
        
    def update_reward(self, phi, r, task_index, exact=False):
        w = np.asarray(self.fit_w[task_index])
        # Ensure column vector shape (n_features, 1)
        w = w.reshape(-1, 1)
        phi = np.asarray(phi).reshape(w.shape)
        r_fit = float(np.sum(phi * w))
        self.fit_w[task_index] = (w + self.alpha_w * (r - r_fit) * phi).reshape(w.shape)
        # validate reward
        r_true = float(np.sum(phi * np.asarray(self.true_w[task_index]).reshape(w.shape)))
        if exact and not np.allclose(r, r_true):
            raise Exception('sampled reward {} != linear reward {} - please check task {}!'.format(
                r, r_true, task_index))

    
    def GPE_w(self, state, policy_index, w):
        """
        Implements generalized policy evaluation according to [1]. In summary, this uses the
        learned reward parameters of one task and successor features of a policy to estimate the Q-values of 
        the policy if it were executed in that task.
        
        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        policy_index : integer
            the index of the task whose policy to evaluate
        w : numpy array
            reward parameters of the task in which to evaluate the policy
            
        Returns
        -------
        np.ndarray : the estimated Q-values of shape [n_batch, n_actions], where
            n_batch is the number of states in the state argument
            n_actions is the number of actions in the MDP            
        """
        w_arr = np.asarray(w).reshape(-1,1)
        psi = self.get_successor(state, policy_index)  # expected np.ndarray [B, A, F]
        # matrix multiply: [B, A, F] @ [F,1] -> [B, A, 1]
        q = np.matmul(psi, w_arr)
        q = q.reshape(q.shape[0], q.shape[1])  # [B, A]
        return q
        
    def GPE(self, state, policy_index, task_index):
        """
        Implements generalized policy evaluation according to [1]. In summary, this uses the
        learned reward parameters of one task and successor features of a policy to estimate the Q-values of 
        the policy if it were executed in that task.
        
        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        policy_index : integer
            the index of the task whose policy to evaluate
        task_index : integer
            the index of the task (e.g. reward) to use to evaluate the policy
            
        Returns
        -------
        np.ndarray : the estimated Q-values of shpae [n_batch, n_actions], where
            n_batch is the number of states in the state argument
            n_actions is the number of actions in the MDP            
        """
        return self.GPE_w(state, policy_index, self.fit_w[task_index])
    
    def GPI_w(self, state, w):
        """
        Implements generalized policy improvement according to [1]. 
        
        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        w : numpy array
            the reward parameters of the task to control
        
        Returns
        -------
        np.ndarray : the maximum Q-values computed by GPI for selecting actions
        of shape [n_batch, n_tasks, n_actions], where:
            n_batch is the number of states in the state argument
            n_tasks is the number of tasks
            n_actions is the number of actions in the MDP 
        np.ndarray : the tasks that are active in each state of state_batch in GPi
        """
        psi = self.get_successors(state)
        w_arr = np.asarray(w).reshape(-1,1)  # [F,1]
        # compute q: [B, n_tasks, n_actions, 1] -> squeeze last dim
        q = np.matmul(psi, w_arr)  # broadcasting matmul
        q = np.squeeze(q, axis=-1)  # [B, n_tasks, n_actions]
        # choose best task per state: max over actions then argmax over tasks
        task = np.argmax(np.max(q, axis=2), axis=1)  # shape [B]
        return q, task

    
    def GPI(self, state, task_index, update_counters=False):
        """
        Implements generalized policy improvement according to [1]. 
        
        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        task_index : integer
            the index of the task in which the GPI action will be used
        update_counters : boolean
            whether or not to keep track of which policies are active in GPI
        
        Returns
        -------
        np.ndarray : the maximum Q-values computed by GPI for selecting actions
        of shape [n_batch, n_tasks, n_actions], where:
            n_batch is the number of states in the state argument
            n_tasks is the number of tasks
            n_actions is the number of actions in the MDP 
        np.ndarray : the tasks that are active in each state of state_batch in GPi
        """
        q, task = self.GPI_w(state, self.fit_w[task_index])
        if update_counters:
            self.gpi_counters[task_index][task] += 1
        return q, task
    
    def GPI_usage_percent(self, task_index):
        """
        Counts the number of times that actions were transferred from other tasks.
        
        Parameters
        ----------
        task_index : integer
            the index of the task
        
        Returns
        -------
        float : the (normalized) number of actions that were transferred from other
            tasks in GPi.
        """
        counts = self.gpi_counters[task_index]        
        return 1. - (float(counts[task_index]) / np.sum(counts))