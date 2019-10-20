import rlkit.torch.pytorch_util as ptu

def move_to_cpu():
    ptu.set_gpu_mode(False)

def move_to_cuda(config):
    if config['use_gpu']:
        if 'cuda_device' in config:
            cuda_device = config['cuda_device']
        else:
            cuda_device = 0
        ptu.set_gpu_mode(True, cuda_device)

def copy_pop_to_ind(networks_pop, networks_ind):
    for key in networks_pop:
        state_dict = networks_pop[key].state_dict()
        networks_ind[key].load_state_dict(state_dict)
        networks_ind[key].eval()

def copy_policie_to_cpu(policy_cpu, policy_gpu):
    policy_dict = policy_gpu.state_dict()
    for key, val in policy_dict.items():
        policy_dict[key] = val.cpu()
    policy_cpu.load_state_dict(policy_dict)
    policy_cpu = policy_cpu.cpu()
    policy_cpu.eval()
    return policy_cpu

def copy_network(network_to, network_from, force_cpu=False):
    policy_dict = policy_gpu.state_dict()
    if force_cpu:
        for key, val in policy_dict.items():
            policy_dict[key] = val.cpu()
    policy_cpu.load_state_dict(policy_dict)
    if force_cpu:
        policy_cpu = policy_cpu.cpu()
    policy_cpu.eval()
    return policy_cpu
