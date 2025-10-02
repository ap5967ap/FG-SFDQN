import numpy as np
import random

from agents.agent import Agent
import time

class DQN(Agent):
    
    def __init__(self, model_lambda, buffer, *args, target_update_ev=1000, test_epsilon=0.03, **kwargs):
        """
        Creates a new DQN agent that supports universal value function approximation (UVFA).
        
        Parameters
        ----------
        model_lambda : function
            returns a keras Model instance
        buffer : ReplayBuffer
            a replay buffer that implements randomized experience replay
        target_update_ev : integer
            how often to update the target network (defaults to 1000)  (alpha / beta)
        test_epsilon : float
            the exploration parameter for epsilon greedy used during testing 
            (defaults to 0.03 as in the paper)
        """
        super(DQN, self).__init__(*args, **kwargs)
        self.model_lambda = model_lambda
        self.buffer = buffer
        self.target_update_ev = target_update_ev
        self.test_epsilon = test_epsilon
    
    def reset(self):
        Agent.reset(self)
        self.Q = self.model_lambda()
        self.target_Q = self.model_lambda()
        self.target_Q.set_weights(self.Q.get_weights())
        self.buffer.reset()
        self.updates_since_target_updated = 0
        
    def get_Q_values(self, s, s_enc):
        return self.Q.predict_on_batch(s_enc)
    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        
        # remember this experience
        self.buffer.append(s_enc, a, r, s1_enc, gamma)
        
        # sample experience at random
        batch = self.buffer.replay()
        if batch is None: return
        states, actions, rewards, next_states, gammas = batch
        n_batch = self.buffer.n_batch
        indices = np.arange(n_batch)
        rewards = rewards.flatten()

        # main update
        next_actions = np.argmax(self.Q.predict_on_batch(next_states), axis=1)
        targets = self.Q.predict_on_batch(states)
        targets[indices, actions] = rewards + gammas * self.target_Q.predict_on_batch(next_states)[indices, next_actions]
        self.Q.train_on_batch(states, targets)
        
        # target update
        self.updates_since_target_updated += 1
        if self.updates_since_target_updated >= self.target_update_ev:
            self.target_Q.set_weights(self.Q.get_weights())
            self.updates_since_target_updated = 0
    
    def train(self, train_tasks, n_samples, viewers=None, n_view_ev=None, test_tasks=[], n_test_ev=1000):
        if viewers is None: 
            viewers = [None] * len(train_tasks)
        
        
        # add tasks
        self.reset()
        for train_task in train_tasks:
            self.add_training_task(train_task)
        print(train_task)    
        # train each one
        return_data = []
        for index, (train_task, viewer) in enumerate(zip(train_tasks, viewers)):
            self.set_active_training_task(index)
            for t in range(n_samples):
                
                # train
                self.next_sample(viewer, n_view_ev)
                
                # test
                if t % n_test_ev == 0:
                    Rs = []
                    for test_task in test_tasks:
                        R = self.test_agent(test_task)
                        Rs.append(R)
                    avg_R = np.mean(Rs)
                    return_data.append(avg_R)
                    print('test performance: {}'.format('\t'.join(map('{:.4f}'.format, Rs))))
        return return_data
    
    def get_test_action(self, s_enc):
        if random.random() <= self.test_epsilon:
            a = random.randrange(self.n_actions)
        else:
            q = self.get_Q_values(s_enc, s_enc)
            a = np.argmax(q)
        return a
            
    def test_agent(self, task):
        R = 0.
        s = task.initialize()
        s_enc = self.encoding(s)
        for _ in range(self.T):
            a = self.get_test_action(s_enc)
            s1, r, done = task.transition(a)
            s1_enc = self.encoding(s1)
            s, s_enc = s1, s1_enc
            R += r
            if done:
                break
        return R


    def test_agent(self, task, return_history=False, visualize=False, pause=0.12, max_steps=None):
        """
        Run one episode with the agent.

        Args:
            task: environment/task object
            return_history (bool): if True, return detailed episode dict
            visualize (bool): if True, render the episode using rich (replay from history)
            pause (float): seconds between frames when visualizing
            max_steps (int|None): override self.T

        Returns:
            If return_history=False: total reward (float).
            If return_history=True: dict with:
                - total_reward (float)
                - steps (int)
                - states (list of states)
                - actions (list of actions)
                - rewards (list of rewards)
        """
        T = max_steps if max_steps is not None else self.T

        total_reward = 0.0
        s = task.initialize()
        s_enc = self.encoding(s)

        states = [s]     
        actions = []
        rewards = []

        for step in range(T):
            a = self.get_test_action(s_enc)
            s1, r, done = task.transition(a)

            actions.append(a)
            rewards.append(r)
            total_reward += r

            states.append(s1)
            s_enc = self.encoding(s1)
            s = s1

            if done:
                break

        episode = {
            "total_reward": total_reward,
            "steps": len(actions),
            "states": states,
            "actions": actions,
            "rewards": rewards
        }

        if visualize:
            self.render_episode_history_rich(episode, task, agent=self, pause=pause)

        if return_history:
            return episode
        return total_reward

    
    @staticmethod
    def render_episode_history_rich(episode, task, agent=None, pause=0.12, style_map=None):
        """
        Replay an episode (states/actions/rewards) using rich Live.
        - episode: dict as returned by test_agent
        - task: the env/task (needed for maze layout & shape_ids)
        - agent: optional, used for name display
        - pause: seconds between frames
        - style_map: optional override for cell styles (use your DEFAULT_STYLE_MAP)
        """
        from rich.live import Live
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich.console import Console
        import numpy as np
        console = Console()
        DEFAULT_STYLE_MAP = {
            ' ': ("  ", "on grey93"),   # empty cell (light background)
            'X': ("██", "bold on grey11"), # wall (dark block)
            'G': ("G ", "bold on green"),
            '_': ("S ", "bold on blue"),   # start maybe
            '1': ("1 ", "bold on cyan"),
            '2': ("2 ", "bold on magenta"),
            '3': ("3 ", "bold on yellow"),
            # fallback
        }
        style_map = style_map or DEFAULT_STYLE_MAP

        states = episode["states"]
        rewards = episode["rewards"]
        total_steps = episode["steps"]
        def _cell_text(ch, style_map=DEFAULT_STYLE_MAP):
            """
            Return a rich Text for a single cell char.
            """
            AGENT_MARK = ("A ", "bold on red")
            if ch == 'A':
                text = Text(AGENT_MARK[0])
                text.stylize(AGENT_MARK[1])
                return text

            rep, style = style_map.get(ch, (f"{ch} ", "on grey82"))
            t = Text(rep)
            t.stylize(style)
            return t

        def _grid_to_table(grid, style_map=DEFAULT_STYLE_MAP):
            """
            Convert a 2D array-like of single-character strings to a rich Table for display.
            """
            rows, cols = grid.shape
            table = Table.grid(padding=0)
            # create columns
            for _ in range(cols):
                table.add_column(no_wrap=True, width=2)
            for r in range(rows):
                cells = []
                for c in range(cols):
                    ch = str(grid[r, c])
                    cells.append(_cell_text(ch, style_map))
                table.add_row(*cells)
            return table
        # states length is steps+1 (initial + after each action)
        # We'll iterate over states index i = 0..steps (show initial + after each action)
        with Live(refresh_per_second=10, console=console, screen=False) as live:
            for i, state in enumerate(states):
                # state is expected to be (position, collected)
                position, collected = state

                # copy maze and render collected items as empty
                grid = np.copy(task.maze).astype(str)

                # handle both shape_ids formats: {(r,c): idx} OR {idx: (r,c)}
                if task.shape_ids:
                    first_key = next(iter(task.shape_ids.keys()))
                    if isinstance(first_key, tuple):
                        # {(r,c): idx}
                        for coords, idx in task.shape_ids.items():
                            if isinstance(idx, (int, np.integer)) and idx < len(collected) and collected[idx]:
                                grid[coords] = ' '
                    else:
                        # {idx: (r,c)}
                        for idx, coords in task.shape_ids.items():
                            if idx < len(collected) and collected[idx]:
                                grid[coords] = ' '

                grid[position] = 'A'

                

                # build table for grid (reuse your helper)
                grid_table = _grid_to_table(grid, style_map)

                # compute last reward and total so far (rewards correspond to transitions: rewards[j] -> states[j+1])
                last_reward = rewards[i-1] if i > 0 and len(rewards) >= i else 0
                total_so_far = sum(rewards[:i]) if i > 0 else 0
                collected_count = sum(1 for c in collected if c)

                info = Text()
                info.append(f" Step: {i}/{total_steps}\n", style="bold")
                info.append(f" Collected: {collected_count}/{len(collected)}\n")
                info.append(f" Last Reward: {last_reward}\n", style="bold yellow")
                info.append(f" Total Reward: {total_so_far}\n", style="bold green")
                if agent is not None and hasattr(agent, "name"):
                    info.append(f" Agent: {agent.name}\n")

                panel = Panel(info, title="Episode Info", padding=(1, 2))

                outer = Table.grid(expand=False)
                outer.add_column()
                outer.add_column()
                outer.add_row(grid_table, panel)

                live.update(outer)

                # if we've just displayed the final state (i == steps), break after showing the totals
                if i == total_steps:
                    done_panel = Panel(Text(f"Episode finished in {total_steps} steps.\nTotal reward: {episode['total_reward']}\n", style="bold green"), title="Done", padding=(1,2))
                    live.update(Table.grid().add_row(grid_table, done_panel))
                    break

                time.sleep(pause)

        console.print("--- Replay complete ---", style="bold magenta")
