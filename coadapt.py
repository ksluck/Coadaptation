from RL.soft_actor import SoftActorCritic
from DO.pso_batch import PSO_batch
from DO.pso_sim import PSO_simulation
from Environments import evoenvs as evoenvs
import utils
import time
from RL.evoreplay import EvoReplayLocalGlobalStart
import numpy as np
import os
import csv

def select_design_opt_alg(alg_name):
    if alg_name == "pso_batch":
        return PSO_batch
    elif alg_name == "pso_sim":
        return PSO_simulation
    else:
        print("Design Optimization method not found.")
        exit(0)

def select_environment(env_name):
    if env_name == 'HalfCheetah':
        return evoenvs.HalfCheetahEnv
    else:
        print("Environment class not found.")
        exit(0)

def select_rl_alg(rl_name):
    if rl_name == 'SoftActorCritic':
        return SoftActorCritic
    else:
        print('RL method not fund.')
        exit(0)

class Coadaptation(object):

    def __init__(self, config):

        self._config = config
        utils.move_to_cuda(self._config)

        self._episode_length = self._config['pipeline_config']['algo_params']['num_steps_per_epoch']
        self._reward_scale = self._config['pipeline_config']['algo_params']['reward_scale']

        self._env_class = select_environment(self._config['env']['env_name'])
        self._env = evoenvs.HalfCheetahEnv(config=self._config)

        self._replay = EvoReplayLocalGlobalStart(self._env,
            max_replay_buffer_size_species=int(1e6),
            max_replay_buffer_size_population=int(1e7))

        self._networks = {
            'individual' : SoftActorCritic.create_networks(env=self._env, config=config),
            'population' : SoftActorCritic.create_networks(env=self._env, config=config),
        }

        self._rl_alg_class = select_rl_alg(self._config['rl_method'])
        self._rl_alg = SoftActorCritic(config=self._config, env=self._env , replay=self._replay, networks=self._networks)
        self._rl_alg_class = SoftActorCritic

        self._do_alg_class = select_design_opt_alg(self._config['design_optim_method'])
        self._do_alg = self._do_alg_class(config=self._config, replay=self._replay, env=self._env)

        utils.move_to_cpu()
        self._policy_cpu = self._rl_alg_class.get_policy_network(SoftActorCritic.create_networks(env=self._env, config=config))
        utils.move_to_cuda(self._config)

        self._last_single_iteration_time = 0
        self._design_counter = 0
        self._episode_counter = 0
        self._data_design_type = 'Initial'

    def initialize_episode(self):
        utils.copy_pop_to_ind(networks_pop=self._networks['population'], networks_ind=self._networks['individual'])
        # self._rl_alg.initialize_episode(init_networks = True, copy_from_gobal = True)
        self._rl_alg.episode_init()

        self._data_rewards = []
        self._episode_counter = 0


    def single_iteration(self):
        print("Time for one iteration: {}".format(time.time() - self._last_single_iteration_time))
        self._last_single_iteration_time = time.time()
        self._replay.set_mode("species")
        self.collect_training_experience()
        # TODO Change here to train global only after five designs
        train_pop = self._design_counter > 3
        if self._episode_counter >= self._config['initial_episodes']:
            self._rl_alg.single_train_step(train_ind=True, train_pop=train_pop)
        self._episode_counter += 1

    def collect_training_experience(self):
        state = self._env.reset()
        nmbr_of_steps = 0
        done = False

        if self._episode_counter < self._config['initial_episodes']:
            policy_gpu_ind = self._rl_alg_class.get_policy_network(self._networks['population'])
        else:
            policy_gpu_ind = self._rl_alg_class.get_policy_network(self._networks['individual'])
        self._policy_cpu = utils.copy_policie_to_cpu(policy_cpu=self._policy_cpu, policy_gpu=policy_gpu_ind)

        utils.move_to_cpu()

        while not(done) and nmbr_of_steps <= self._episode_length:
            nmbr_of_steps += 1
            action, _ = self._policy_cpu.get_action(state)
            new_state, reward, done, info = self._env.step(action)
            # TODO this has to be fixed _variant_spec
            reward = reward * self._reward_scale
            terminal = np.array([done])
            reward = np.array([reward])
            self._replay.add_sample(observation=state, action=action, reward=reward, next_observation=new_state,
                           terminal=terminal)
            state = new_state
        self._replay.terminate_episode()
        utils.move_to_cuda(self._config)

    def execute_policy(self):
        state = self._env.reset()
        done = False
        reward_ep = 0.0
        reward_original = 0.0
        action_cost = 0.0
        nmbr_of_steps = 0

        if self._episode_counter < self._config['initial_episodes']:
            policy_gpu_ind = self._rl_alg_class.get_policy_network(self._networks['population'])
        else:
            policy_gpu_ind = self._rl_alg_class.get_policy_network(self._networks['individual'])
        self._policy_cpu = utils.copy_policie_to_cpu(policy_cpu=self._policy_cpu, policy_gpu=policy_gpu_ind)

        utils.move_to_cpu()

        while not(done) and nmbr_of_steps <= self._episode_length:
            nmbr_of_steps += 1
            action, _ = self._policy_cpu.get_action(state, deterministic=True)
            new_state, reward, done, info = self._env.step(action)
            action_cost += info['orig_action_cost']
            reward_ep += float(reward)
            reward_original += float(info['orig_reward'])
            state = new_state
        utils.move_to_cuda(self._config)
        # Do something here to log the results
        self._data_rewards.append(reward_ep)

    def save_networks(self):
        checkpoints_pop = {}
        for key in self._networks['population']:
            checkpoints_pop[key] = checkpoints_pop[key].state_dict()

        checkpoints_ind = {}
        for key in self._networks['individual']:
            checkpoints_ind[key] = checkpoints_ind[key].state_dict()

        checkpoint = {
            'population' : checkpoints_pop,
            'individual' : checkpoints_ind,
        }
        file_path = os.path.join(self._config['data_folder_experiment'], 'checkpoints')
        torch.save(checkpoint, os.path.join(file_path, 'checkpoint_{}.chk'.format(self._counter)))

    def load_networks(self):
        pass

    def save_logged_data(self):
        file_path = self._config['data_folder_experiment']
        current_design = self._env.get_current_design()

        with open(
            os.path.join(file_path,
                'data_design_{}.csv'.format(self._design_counter)
                ), 'w') as fd:
            cwriter = csv.writer(fd)
            cwriter.writerow(['Design Type:', self._data_design_type])
            cwriter.writerow(current_design)
            cwriter.writerow(self._data_rewards)

    def run(self):
        iterations_init = self._config['iterations_init']
        iterations = self._config['iterations']
        design_cycles = self._config['design_cycles']
        exploration_strategy = self._config['exploration_strategy']

        self._intial_design_loop(iterations_init)
        self._training_loop(iterations, design_cycles, exploration_strategy)

    def _training_loop(self, iterations, design_cycles, exploration_strategy):
        self.initialize_episode()
        # TODO fix the following
        initial_state = self._env._env.reset()

        self._data_design_type = 'Optimized'

        optimized_params = self._env.get_random_design()
        q_network = self._rl_alg_class.get_q_network(self._networks['population'])
        policy_network = self._rl_alg_class.get_policy_network(self._networks['population'])
        optimized_params = self._do_alg.optimize_design(design=optimized_params, q_network=q_network, policy_network=policy_network)
        optimized_params = list(optimized_params)

        for i in range(design_cycles):
            self._design_counter += 1
            self._env.set_new_design(optimized_params)

            # Reinforcement Learning
            for _ in range(iterations):
                self.single_iteration()
                self.execute_policy()
                self.save_logged_data()

            # Design Optimization
            if i % 2 == 1:
                self._data_design_type = 'Optimized'
                q_network = self._rl_alg_class.get_q_network(self._networks['population'])
                policy_network = self._rl_alg_class.get_policy_network(self._networks['population'])
                optimized_params = self._do_alg.optimize_design(design=optimized_params, q_network=q_network, policy_network=policy_network)
                optimized_params = list(optimized_params)
            else:
                self._data_design_type = 'Random'
                optimized_params = self._env.get_random_design()
                optimized_params = list(optimized_params)
            self.initialize_episode()

    def _intial_design_loop(self, iterations):
        self._data_design_type = 'Initial'
        for params in self._env.init_sim_params:
            self._design_counter += 1
            self._env.set_new_design(params)
            self.initialize_episode()

            # Reinforcement Learning
            for _ in range(iterations):
                self.single_iteration()
                self.execute_policy()
                self.save_logged_data()
