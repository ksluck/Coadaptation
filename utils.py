import rlkit.torch.pytorch_util as ptu
import cv2
import os
from shutil import copyfile, move
import time
import numpy as np

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

def copy_network(network_to, network_from, config, force_cpu=False):
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
    else:
        move_to_cuda(config)
    network_to.load_state_dict(network_from_dict)
    if force_cpu:
        network_to = network_to.to('cpu')
    else:
        network_to.to(ptu.device)
    network_to.eval()
    return network_to

class BestEpisodesVideoRecorder(object):
    def __init__(self, path=None, max_videos=1):
        self._vid_path = '/tmp/videos' if path is None else path

        self._folder_counter = 0
        self._keep_n_best = max(max_videos, 1)
        self._record_evy_n_episodes = 5

        self._frame_width = 200
        self._frame_height = 200
        self._fps_per_frame = 0

        self.increase_folder_counter()
        self._create_vid_stream()
        self._time_start = time.time()

    def _episodic_reset(self):
        self._current_episode_reward = 0
        self._did_at_least_one_step = False
        self._step_counter = 1

    def reset_recorder(self):
        self._episode_counter = 0
        self._episodic_rewards = [-float('inf')] * self._keep_n_best
        self._episodic_reset()


    def increase_folder_counter(self):
        self._current_vid_path = os.path.join(self._vid_path, str(self._folder_counter))
        self.reset_recorder()
        self._folder_counter += 1

    def step(self, env, state, reward, done):
        if self._episode_counter % self._record_evy_n_episodes == 0:
            self._current_episode_reward += reward
            env.camera_adjust()
            frame = env.render_camera_image((self._frame_width, self._frame_height))
            frame = frame * 255
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._vid_writer.write(frame)
            self._did_at_least_one_step = True
        proc_time = (time.time() - self._time_start)*1000
        proc_time = 1000/proc_time
        self._time_start = time.time()
        self._fps_per_frame += proc_time
        self._step_counter += 1

    def _do_video_file_rotation(self):
        for idx, elem in enumerate(self._episodic_rewards):
            if idx > 1:
                try:
                    move(os.path.join(self._current_vid_path, 'video_{}.avi'.format(idx-1)), os.path.join(self._current_vid_path, 'video_{}.avi'.format(idx-2)))
                except:
                    pass
            if self._current_episode_reward < elem:
                self._episodic_rewards = self._episodic_rewards[1:idx] + [self._current_episode_reward] + self._episodic_rewards[idx:]
                copyfile(os.path.join(self._current_vid_path, 'current_video.avi'), os.path.join(self._current_vid_path, 'video_{}.avi'.format(idx-1)))
                break
            # Will only be true in last iteration and only be hit if last element is to be moved
            if idx == len(self._episodic_rewards)-1:
                try:
                    move(os.path.join(self._current_vid_path, 'video_{}.avi'.format(idx)), os.path.join(self._current_vid_path, 'video_{}.avi'.format(idx-1)))
                except:
                    pass
                self._episodic_rewards = self._episodic_rewards[1:] + [self._current_episode_reward]
                copyfile(os.path.join(self._current_vid_path, 'current_video.avi'), os.path.join(self._current_vid_path, 'video_{}.avi'.format(idx)))


    def reset(self, env, state, reward, done):
        # final processing of data from previous episode
        if self._episode_counter % self._record_evy_n_episodes == 0:
            env.camera_adjust()
            self._vid_writer.release()
            if not os.path.exists(self._current_vid_path):
                os.makedirs(self._current_vid_path)
            if self._did_at_least_one_step and min(self._episodic_rewards) < self._current_episode_reward:
                self._do_video_file_rotation()
            print('Average FPS of last episode: {}'.format(self._fps_per_frame/self._step_counter))

        self._episode_counter += 1
        self._episodic_reset()
        # set up everything for this episode if we record
        if self._episode_counter % self._record_evy_n_episodes == 0:
            self._create_vid_stream()
            frame = env.render_camera_image((self._frame_width, self._frame_height))
            frame = frame * 255
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._vid_writer.write(frame)

            self._time_start = time.time()
            self._fps_per_frame = 0
            self._step_counter = 1

    def _create_vid_stream(self):
        if not os.path.exists(self._current_vid_path):
            os.makedirs(self._current_vid_path)
        self._vid_writer = cv2.VideoWriter(os.path.join(self._current_vid_path, 'current_video.avi'),
            cv2.VideoWriter_fourcc('M','J','P','G'), 30, (self._frame_width, self._frame_height))
