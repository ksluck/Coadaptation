from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
import numpy as np

class EvoReplayLocalGlobalStart(ReplayBuffer):
    def __init__(self, env, max_replay_buffer_size_species, max_replay_buffer_size_population):
        self._species_buffer = EnvReplayBuffer(env=env, max_replay_buffer_size=max_replay_buffer_size_species)
        self._population_buffer = EnvReplayBuffer(env=env, max_replay_buffer_size=max_replay_buffer_size_population)
        self._init_state_buffer = EnvReplayBuffer(env=env, max_replay_buffer_size=max_replay_buffer_size_population)
        self._env = env
        self._max_replay_buffer_size_species = max_replay_buffer_size_species
        self._mode = "species"
        self._ep_counter = 0
        self._expect_init_state = True
        print("Use EvoReplayLocalGlobalStart replay buffer")

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        """
        Add a transition tuple.
        """
        if self._mode == "species":
            self._species_buffer.add_sample(observation=observation, action=action, reward=reward, next_observation=next_observation,
                           terminal=terminal, env_info={}, **kwargs)
            if self._expect_init_state:
                self._init_state_buffer.add_sample(observation=observation, action=action, reward=reward, next_observation=next_observation,
                               terminal=terminal, env_info={}, **kwargs)
                self._init_state_buffer.terminate_episode()
                self._expect_init_state = False

            if self._ep_counter >= 0:
                self._population_buffer.add_sample(observation=observation, action=action, reward=reward, next_observation=next_observation,
                           terminal=terminal, env_info={}, **kwargs)

    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        if self._mode == "species":
            self._species_buffer.terminate_episode()
            self._population_buffer.terminate_episode()
            self._ep_counter += 1
            self._expect_init_state = True

    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        if self._mode == "species":
            return self._species_buffer.num_steps_can_sample(**kwargs)
        elif  self._mode == "population":
            return self._population_buffer.num_steps_can_sample(**kwargs)
        else:
            pass

    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        if self._mode == "species":
            # return self._species_buffer.random_batch(batch_size)
            species_batch_size = int(np.floor(batch_size * 0.9))
            pop_batch_size = int(np.ceil(batch_size * 0.1))
            pop = self._population_buffer.random_batch(pop_batch_size)
            spec = self._species_buffer.random_batch(species_batch_size)
            for key, item in pop.items():
                pop[key] = np.concatenate([pop[key], spec[key]], axis=0)
            return pop
        elif self._mode == "population":
            return self._population_buffer.random_batch(batch_size)
        elif self._mode == "start":
            return self._init_state_buffer.random_batch(batch_size)
        else:
            pass

    def set_mode(self, mode):
        if mode == "species":
            self._mode = mode
        elif mode == "population":
            self._mode = mode
        elif mode == "start":
            self._mode = mode
        else:
            print("No known mode :(")

    def reset_species_buffer(self):
        self._species_buffer = EnvReplayBuffer(env = self._env, max_replay_buffer_size=self._max_replay_buffer_size_species)
        self._ep_counter = 0
