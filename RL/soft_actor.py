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
        """ Bascally a wrapper class for SAC from rlkit.

        Args:
            config: Configuration dictonary
            env: Environment
            replay: Replay buffer
            networks: dict containing two sub-dicts, 'individual' and 'population'
                which contain the networks.

        """
        super().__init__(config, env, replay, networks)

        self._variant_pop = config['rl_algorithm_config']['algo_params_pop']
        self._variant_spec = config['rl_algorithm_config']['algo_params']

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
        """ Initializations to be done before the first episode.

        In this case basically creates a fresh instance of SAC for the
        individual networks and copies the values of the target network.
        """
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
        """ A single trianing step.

        Args:
            train_ind: Boolean. If true the individual networks will be trained.
            train_pop: Boolean. If true the population networks will be trained.
        """
        if train_ind:
          self._algorithm_ind.num_updates_per_train_call = self._variant_spec['num_updates_per_epoch']
          self._algorithm_ind._try_to_train()

        if train_pop:
          self._algorithm_pop.num_updates_per_train_call = self._variant_pop['num_updates_per_epoch']
          self._algorithm_pop._try_to_train()

    @staticmethod
    def create_networks(env, config):
        """ Creates all networks necessary for SAC.

        These networks have to be created before instantiating this class and
        used in the constructor.

        TODO: Maybe this should be reworked one day...

        Args:
            config: A configuration dictonary.

        Returns:
            A dictonary which contains the networks.
        """
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        net_size = config['rl_algorithm_config']['net_size']
        hidden_sizes = [net_size] * config['rl_algorithm_config']['network_depth']
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
        """ Returns the q network from a dict of networks.

        This method extracts the q-network from the dictonary of networks
        created by the function create_networks.

        Args:
            networks: Dict containing the networks.

        Returns:
            The q-network as torch object.
        """
        return networks['qf']

    @staticmethod
    def get_policy_network(networks):
        """ Returns the policy network from a dict of networks.

        This method extracts the policy network from the dictonary of networks
        created by the function create_networks.

        Args:
            networks: Dict containing the networks.

        Returns:
            The policy network as torch object.
        """
        return networks['policy']
