

sac_pso_batch = {
    'name' : 'Experiment 1: PSO Batch',
    'data_folder' : 'data_exp_sac_pso_batch',
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
    'design_optim_method' : 'pso_batch',
    'rl_method' : 'SoftActorCritic',
    'rl_algorithm_config' : dict(
        algo_params=dict(
            num_epochs=int(1),
            num_steps_per_epoch=1000,
            num_steps_per_eval=1,
            num_updates_per_env_step=10,
            num_updates_per_epoch=1000,
            batch_size=256,
            max_path_length=1000,
            discount=0.99,
            reward_scale=1.0,

            soft_target_tau=0.005,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            save_environment=False,
            collection_mode='batch',
            ),
        algo_params_pop=dict(
            num_epochs=int(1),
            num_steps_per_epoch=1,
            num_steps_per_eval=1,
            num_updates_per_env_step=10,
            num_updates_per_epoch=250,
            batch_size=256,
            max_path_length=1,
            discount=0.99,
            reward_scale=1.0,

            soft_target_tau=0.005,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            save_environment=False,
            collection_mode='batch',
            ),
        net_size=200,
        network_depth=3,
        copy_from_gobal=True,
    ),
    'env' : dict(
        env_name='HalfCheetah',
        render=False,
        record_video=True,
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
    'rl_method' : 'SoftActorCritic',
    'rl_algorithm_config' : dict(
        algo_params=dict(
            num_epochs=int(1),
            num_steps_per_epoch=1000,
            num_steps_per_eval=1,
            num_updates_per_env_step=10,
            num_updates_per_epoch=1000,
            batch_size=256,
            max_path_length=1000,
            discount=0.99,
            reward_scale=1.0,

            soft_target_tau=0.005,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            save_environment=False,
            collection_mode='batch',
            ),
        algo_params_pop=dict(
            num_epochs=int(1),
            num_steps_per_epoch=1,
            num_steps_per_eval=1,
            num_updates_per_env_step=10,
            num_updates_per_epoch=250,
            batch_size=256,
            max_path_length=1,
            discount=0.99,
            reward_scale=1.0,

            soft_target_tau=0.005,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            save_environment=False,
            collection_mode='batch',
            ),
        net_size=200,
        network_depth=3,
        copy_from_gobal=True,
    ),
    'env' : dict(
        env_name='HalfCheetah',
        render=True,
    ),
    }

config_dict = {
    'sac_pso_batch' : sac_pso_batch,
    'sac_pso_sim' : sac_pso_sim,
    }
