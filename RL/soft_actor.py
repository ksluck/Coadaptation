from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp
import numpy as np
from .rl_algorithm import RL_algorithm
from rlkit.torch.sac.sac import SoftActorCritic as SoftActorCritic_rlkit
import rlkit.torch.pytorch_util as ptu
import torch

# networks = {individual:, population:}
class SoftActorCritic(RL_algorithm):

    def __init__(self, config, env, replay, networks):
        super().__init__(config, env, replay, networks)

        self._variant_pop = config['pipeline_config']['algo_params_pop']
        self._variant_spec = config['pipeline_config']['algo_params']

        self._variant_pop['replay_buffer'] = self._replay
        self._variant_spec['replay_buffer'] = self._replay

        self._ind_qf = networks['individual']['qf']
        self._ind_vf = networks['individual']['vf']
        self._ind_policy = networks['individual']['policy']

        self._pop_qf = networks['population']['qf']
        self._pop_vf = networks['population']['vf']
        self._pop_policy = networks['population']['policy']

        self._alt_alpha = 0.01
        self._algorithm_ind = SoftActorCritic_rlkit(
            env=self._env,
            policy=self._ind_policy,
            qf=self._ind_qf,
            vf=self._ind_vf,
            use_automatic_entropy_tuning = False,
            alt_alpha = self._alt_alpha,
            **self._variant_spec
        )

        self._algorithm_pop = SoftActorCritic_rlkit(
            env=self._env,
            policy=self._pop_policy,
            qf=self._pop_qf,
            vf=self._pop_vf,
            use_automatic_entropy_tuning = False,
            alt_alpha = self._alt_alpha,
            **self._variant_pop
        )

        self._algorithm_ind.to(ptu.device)
        self._algorithm_pop.to(ptu.device)

    def episode_init(self):
        self._algorithm_ind = SoftActorCritic_rlkit(
            env=self._env,
            policy=self._ind_policy,
            qf=self._ind_qf,
            vf=self._ind_vf,
            use_automatic_entropy_tuning = False,
            alt_alpha = self._alt_alpha,
            **self._variant_spec
        )
        # We have only to do this becasue the version of rlkit which we use
        # creates internally a target network
        vf_dict = self._algorithm_pop.target_vf.state_dict()
        self._algorithm_ind.target_vf.load_state_dict(vf_dict)
        self._algorithm_ind.target_vf.eval()
        self._algorithm_ind.to(ptu.device)

    def single_train_step(self, train_ind=True, train_pop=False):
        if train_ind:
          self._algorithm_ind.num_updates_per_train_call = self._variant_spec['num_updates_per_epoch']
          self._algorithm_ind._try_to_train()

        if train_pop:
          self._algorithm_pop.num_updates_per_train_call = self._variant_pop['num_updates_per_epoch']
          self._algorithm_pop._try_to_train()

    @staticmethod
    def create_networks(env, config):
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        net_size = config['pipeline_config']['net_size']
        hidden_sizes = [net_size] * config['pipeline_config']['network_depth']
        # hidden_sizes = [net_size, net_size, net_size]
        qf = FlattenMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        )
        vf = FlattenMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim,
            output_size=1,
        )
        policy = TanhGaussianPolicy(
            hidden_sizes=hidden_sizes,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        clip_value = 1.0
        for p in qf.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        for p in vf.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        for p in policy.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

        return {'qf' : qf, 'vf' : vf, 'policy' : policy}

    @staticmethod
    def get_q_network(networks):
        return networks['qf']

    def get_policy_network(networks):
        return networks['policy']
