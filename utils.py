import rlkit.torch.pytorch_util as ptu

def move_to_cpu():
    """ Set device to cpu for torch.
    """
    ptu.set_gpu_mode(False)

def move_to_cuda(config):
    """ Set device to CUDA and which GPU, or CPU if not set.

    Args:
        config: Dict containing the configuration.
    """
    if config['use_gpu']:
        if 'cuda_device' in config:
            cuda_device = config['cuda_device']
        else:
            cuda_device = 0
        ptu.set_gpu_mode(True, cuda_device)

def copy_pop_to_ind(networks_pop, networks_ind):
    """ Function used to copy params from pop. networks to individual networks.

    The parameters of all networks in network_ind will be set to the parameters
    of the networks in networks_ind.

    Args:
        networks_pop: Dictonary containing the population networks.
        networks_ind: Dictonary containing the individual networks. These
            networks will be updated.
    """
    for key in networks_pop:
        state_dict = networks_pop[key].state_dict()
        networks_ind[key].load_state_dict(state_dict)
        networks_ind[key].eval()

def copy_policie_to_cpu(policy_cpu, policy_gpu):
    # not used anymore
    policy_dict = policy_gpu.state_dict()
    for key, val in policy_dict.items():
        policy_dict[key] = val.cpu()
    policy_cpu.load_state_dict(policy_dict)
    policy_cpu = policy_cpu.cpu()
    policy_cpu.eval()
    return policy_cpu

def copy_network(network_to, network_from, force_cpu=False):
    """ Copies networks and set them to device or cpu.

    Args:
        networks_to: Netwoks to which we want to copy (destination).
        networks_from: Networks from which we want to copy (source). These
            networks will be changed.
        force_cpu: Boolean, if True the desitnation nateworks will be placed on
            the cpu.  If not the current device will be used.
    """
    network_from_dict = network_from.state_dict()
    if force_cpu:
        for key, val in network_from_dict.items():
            network_from_dict[key] = val.cpu()
    network_to.load_state_dict(network_from_dict)
    if force_cpu:
        network_to = network_to.to('cpu')
    else:
        network_to.to("cuda:" + str(ptu.gpu_id) if ptu._use_gpu else "cpu")
    network_to.eval()
    return network_to
