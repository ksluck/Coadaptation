import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu
import pyswarms as ps
from .design_optimization import Design_Optimization

class PSO_batch(Design_Optimization):

    def __init__(self, config, replay, env):
        self._config = config
        self._replay = replay
        self._env = env

        if 'state_batch_size' in self._config.keys():
            self._state_batch_size = self._config['state_batch_size']
        else:
            self._state_batch_size = 32


    def optimize_design(self, design, q_network, policy_network):
        self._replay.set_mode('start')
        initial_state = self._replay.random_batch(self._state_batch_size)
        initial_state = initial_state['observations']
        design_idxs = self._env.get_design_dimensions()
        # initial_state = initial_state[:,:-len(design)]
        # state_tensor = torch.from_numpy(initial_state).to(device=ptu.device, dtype=torch.float32)

        # initial_design = np.array(self._current_design)
        # initial_design = np.array(design)

        def f_qval(x_input, **kwargs):
            shape = x_input.shape
            cost = np.zeros((shape[0],))
            with torch.no_grad():
                for i in range(shape[0]):
                    x = x_input[i:i+1,:]
                    # X = (
                    #     torch.from_numpy(x)
                    #     .to(device=ptu.device, dtype=torch.float32)
                    #     .contiguous()
                    #     .requires_grad_(False)
                    # )
                    # X_expand = X.expand(self._state_batch_size, -1)
                    # network_input = torch.cat((state_tensor,X_expand), -1)
                    state_batch = initial_state.copy()
                    state_batch[:,design_idxs] = x
                    network_input = torch.from_numpy(state_batch).to(device=ptu.device, dtype=torch.float32)
                    action, _, _, _, _, _, _, _, = policy_network(network_input, deterministic=True)
                    output = q_network(network_input, action)
                    #output = self._vf_pop.forward(input)
                    loss = -output.mean().sum()
                    fval = float(loss.item())
                    cost[i] = fval
            return cost

        lower_bounds = [l for l, _ in self._env.design_params_bounds]
        lower_bounds = np.array(lower_bounds)
        upper_bounds = [u for _, u in self._env.design_params_bounds]
        upper_bounds = np.array(upper_bounds)
        bounds = (lower_bounds, upper_bounds)
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        #options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}
        optimizer = ps.single.GlobalBestPSO(n_particles=700, dimensions=len(design), bounds=bounds, options=options)

        # Perform optimization
        cost, new_design = optimizer.optimize(f_qval, print_step=100, iters=250, verbose=3) #, n_processes=2)
        return new_design
