

sac_pso_batch = {
    'name' : 'Experiment 1: PSO Batch',         # Name of the experiment. Can be used just for reference later
    'data_folder' : 'data_exp_sac_pso_batch',   # Name of the folder in which data generated during the experiments are saved
    'nmbr_random_designs' : 0,                  # Number of random designs to run after the initial design TODO: Not used atm
    'iterations_init' : 300,                    # Number of episodes for all initial designs as provided by the environment class
    'iterations_random': 100,                   # Number of episodes for all random designs TODO: Not used atm
    'iterations' : 100,                         # Number of episodes to run for all designs after the initial
    'design_cycles' : 55,                       # Number of design adaptations after the initial designs
    'state_batch_size' : 32,                    # Size of the batch used during the design optimization process to estimate fitness of design
    'initial_episodes' : 3,                     # Number of initial episodes for each design before we start the training of the individual networks. Useful if steps per episode is low and we want to fill the replay.
    'use_gpu' : True,                           # Use True when GPU should be used for training and inference
    'use_cpu_for_rollout': False,               # TODO: Not used
    'cuda_device': 0,                           # Which cuda device to use. Only relevant if you have more than one GPU
    'exploration_strategy': 'random',           # Type of design exploration to use - we do usually one greedy design optimization and one random selection of a design
    'design_optim_method' : 'pso_batch',        # Which design optimization method to use
    'steps_per_episodes' : 1000,                # Number of steps per episode
    'save_networks' : True,                     # If True networks are checkpointed and saved for each design
    'rl_method' : 'SoftActorCritic',            # Which reinforcement learning method to use.
    'rl_algorithm_config' : dict(               # Dictonary which contains the parameters for the RL algorithm
        algo_params=dict(                           # Parameters for the RL learner for the individual networks
            # num_epochs=int(1),
            # num_steps_per_epoch=1000,
            # num_steps_per_eval=1,
            # num_updates_per_env_step=10,
            # num_updates_per_epoch=1000,
            # batch_size=256,
            discount=0.99,
            reward_scale=1.0,

            soft_target_tau=0.005,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            alpha=0.01,
            ),
        algo_params_pop=dict(                       # Parameters for the RL learner for the individual networks
            # num_epochs=int(1),
            # num_steps_per_epoch=1,
            # num_steps_per_eval=1,
            # num_updates_per_env_step=10,
            # num_updates_per_epoch=250,
            # batch_size=256,
            discount=0.99,
            reward_scale=1.0,

            soft_target_tau=0.005,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            alpha=0.01,
            ),
        net_size=200,                       # Number of neurons in hidden layer
        network_depth=3,                    # Number of hidden layers
        copy_from_gobal=True,               # Shall we pre-initialize the individual network with the global network?
        indiv_updates=1000,                 # Number of training updates per episode for individual networks
        pop_updates=250,                    # Number of training updates per episode for population networks
        batch_size=256,                     # Batch size
    ),
    'env' : dict(                           # Parameters and which environment to use
        env_name='HalfCheetah',             # Name of environment
        render=False,                       # Use True if you want to visualize/render the environment
        record_video=False,                  # Use True if you want to record videos
    ),
    }

sac_pso_sim = {
    'name' : 'Experiment 2: PSO using Simulations',
    'data_folder' : 'data_exp_sac_pso_sim',
    'nmbr_random_designs' : 0,
    'iterations_init' : 300,
    'iterations_random': 100,
    'iterations' : 100,
    'design_cycles' : 55,
    'state_batch_size' : 32,
    'initial_episodes' : 3,
    'use_gpu' : True,
    'use_cpu_for_rollout': False,
    'cuda_device': 0,
    'exploration_strategy': 'random',
    'design_optim_method' : 'pso_sim',
    'save_networks' : True,
    'rl_method' : 'SoftActorCritic',
    'rl_algorithm_config' : dict(
        algo_params=dict(
            # num_epochs=int(1),
            # num_steps_per_epoch=1000,
            # num_steps_per_eval=1,
            # num_updates_per_env_step=10,
            # num_updates_per_epoch=1000,
            # batch_size=256,
            discount=0.99,
            reward_scale=1.0,

            soft_target_tau=0.005,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            alpha=0.01,
            ),
        algo_params_pop=dict(
            # num_epochs=int(1),
            # num_steps_per_epoch=1,
            # num_steps_per_eval=1,
            # num_updates_per_env_step=10,
            # num_updates_per_epoch=250,
            # batch_size=256,
            discount=0.99,
            reward_scale=1.0,

            soft_target_tau=0.005,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            alpha=0.01,
            ),
        net_size=200,
        network_depth=3,
        copy_from_gobal=True,
        indiv_updates=1000,
        pop_updates=250,
        batch_size=256,
    ),
    'env' : dict(
        env_name='HalfCheetah',
        render=True,
        record_video=False,
    ),
    }

config_dict = {
    'sac_pso_batch' : sac_pso_batch,
    'sac_pso_sim' : sac_pso_sim,
    }
