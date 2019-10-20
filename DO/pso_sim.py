import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu
import pyswarms as ps
from .design_optimization import Design_Optimization

class PSO_simulation(Design_Optimization):

    def __init__(self, config, replay, env):
        self._config = config
        self._replay = replay
        self._env = env

        self._episode_length = self._config['pipeline_config']['algo_params']['num_steps_per_epoch']
        self._reward_scale = self._config['pipeline_config']['algo_params']['reward_scale']


    def optimize_design(self, design, q_network, policy_network):
        # Important: We reset the design of the environment. Previous design
        #   will be lost

        def get_reward_for_design(design):
            self._env.set_new_design(design)
            state = self._env.reset()
            reward_episode = []
            done = False
            nmbr_of_steps = 0
            while not(done) and nmbr_of_steps <= self._episode_length:
                nmbr_of_steps += 1
                action, _ = policy_network.get_action(state, deterministic=True)
                new_state, reward, done, info = self._env.step(action)
                reward = reward * self._reward_scale
                reward_episode.append(float(reward))
                state = new_state
            reward_mean = np.mean(reward_episode)
            return reward_mean

        def f_qval(x_input, **kwargs):
            shape = x_input.shape
            cost = np.zeros((shape[0],))
            for i in range(shape[0]):
                x = x_input[i,:]
                reward = get_reward_for_design(x)
                cost[i] = -reward
            return cost

        lower_bounds = [l for l, _ in self._env.design_params_bounds]
        lower_bounds = np.array(lower_bounds)
        upper_bounds = [u for _, u in self._env.design_params_bounds]
        upper_bounds = np.array(upper_bounds)
        bounds = (lower_bounds, upper_bounds)
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=35, dimensions=len(design), bounds=bounds, options=options)

        # Perform optimization
        cost, new_design = optimizer.optimize(f_qval, print_step=100, iters=30, verbose=3)
        return new_design
