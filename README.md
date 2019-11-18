# Fast Evolution through Actor-Critic Reinforcement Learning
This is the official repository providing a refactored implementation of the data-driven design optimization method presented in the paper [**Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning**](https://research.fb.com/publications/data-efficient-co-adaptation-of-morphology-and-behaviour-with-deep-reinforcement-learning/). 
This paper was presented on the Conference on Robot Learning in 2019. 
The website for this project can be found [here](https://sites.google.com/view/drl-coadaptation/home).

At the moment, the repository contains a basic implementation of the proposed algorithm and its baseline. We use particle swarm optimization on the Q-function, which is used as a surrogate function predicting the performance of design candidates and, thus, avoiding the necessity to simulate/evaluate design candidates. The baseline uses also particle swarm optimization but evaluates design candidates in simulation instead.

The current environment provided is Half-Cheetah, using pybullet, for which we have to learn effective movement strategies and the optimal leg lengths, maximizing the performance of the agent.

Additional methods and environments which are shown in the paper will be added over time and the structure of the repository might change in the future.

## Citation
If you use this code in your research, please cite
```
@inproceedings{luck2019coadapt,
  title={Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning},
  author={Luck, Kevin Sebastian and Ben Amor, Heni and Calandra, Roberto},
  booktitle={Conference on Robot Learning},
  year={2019}
}
```

## Acknowledgements of Previous Work
This project would have been harder to implement without the great work of
the developers behind rlkit and pybullet.

The reinforcement learning loop makes extensive use of rlkit, a framework developed
and maintained by Vitchyr Pong. You find this repository [here](https://github.com/vitchyr/rlkit).
We made slight adaptations to the Soft-Actor-Critic algorithm used in this repository.

Tasks were simulated in [PyBullet](https://pybullet.org/wordpress/), the
repository can be found [here](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet).
Adaptations were made to the files found in pybullet_evo to enable the dynamic adaptation
of design parameters during the training process.

### Why do you use an older version of rlkit?
When I started working on this project, the tag v0.1.2 was the newest. There are quite many changes from 0.1.2 to 0.2.0, I will tackle this update in the future ;)

## Installation

Make sure that PyTorch is installed. You find more information here: https://pytorch.org/

First, clone this repository to your local computer as usual.
Then, install the required packages via pip by executing `pip3 install -r requirements.txt`.

The SAC implementation used differs slightly from the original version in
rlkit developed by Vitchyr Pong. For your convenience, I provide a forked repository. However,
all credit for the SAC implementation goes to [Vitchyr Pong](https://github.com/vitchyr/rlkit).

Clone the adapted rlkit with
```bash
git clone https://github.com/ksluck/Coadaptation-rlkit.git
```
Now, set in your terminal the environment variable PYTHONPATH with
```bash
export PYTHONPATH=/path/to/Coadaptation-rlkit/
```
where the folder `/path/to/Coadaptation-rlkit` contains the folder `rlkit`. This enables us
to import rlkit with `import rlkit`.

You may have to set the environmental variable every time you open a new terminal.

## Starting experiments

After setting the environmental variable and installing the packages you can
proceed to run the experiments.
There are two experimental configurations already set up for you in `experiment_configs.py`.
You can execute them with
```bash
python3 main.py sac_pso_batch
```
and
```bash
python3 main.py sac_pso_sim
```

You may change the configs or add new ones. Make sure to add new configurations to
the `config_dict` in `experiment_configs.py`.

## Data logging
If you execute these commands, they will automatically create directories in which
the performance and achieved rewards will be stored. Each experiment creates
a specific folder with the current date/time and a random string as name.
You can find in this folder a copy of the config you executed and one csv file
for each design on which the reinforcement learning algorithm was executed.
Each csv file contains three rows: The type of the design (either 'Initial', 'Optimized' or 'Random');
The design vector; And the subsequent, cumulative rewards for each episode/trial.

The file `ADDVIZFILE` provieds a basic jupyter notebook to visualize the collected
data.
