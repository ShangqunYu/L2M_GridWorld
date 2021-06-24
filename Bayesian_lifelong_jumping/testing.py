import gym
import sys
import time
import os
import json
from gym_jumping_task.envs import JumpTaskEnv
import numpy as np
import pygame
import random
import torch
import click
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import BayesianLifelongRL
from rlkit.torch.sac.agent import Agent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
from rl_alg import BNNdynamics

from gym_env import GymEnv
def experiment(variant):
    envs = {}
    env_id = "jumping-task-v0"
    #breakpoint()
    env = GymEnv(env_id)
    #env = gym.make('jumping-task-v0') #, rendering=True, obstacle_position =17)
    envs[0] = env

    obs_dim = 5
    #Simon:change action dim to 2 for 1 hot mode
    action_dim = 2
    SEED = 50  # initial value, 10 will be added for every iteration
    job_name_mtl = 'results/jumping'
    torch.set_num_threads(5)
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])

    #general model
    dyna = BNNdynamics(obs_dim, action_dim, device = ptu.device,learning_rate=0.0005,weight_out=0.1)
    #set that contains all the task specifc model
    forward_dyna_set = []

    #task specific model
    forward_dyna = BNNdynamics(
        obs_dim, action_dim, device=ptu.device,learning_rate=0.0005,
        deterministic=False,weight_out=0.1)
    #spend the only specific model into the set
    forward_dyna_set.append(forward_dyna)
    #create an agent
    agent = Agent(
        forward_dyna=forward_dyna_set,
        dyna=dyna, action_dim=action_dim, **variant['algo_params'])

    algorithm = BayesianLifelongRL(
        env=envs,
        nets=[agent, forward_dyna_set],
        **variant['algo_params'])

    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()




    for i in range(1000):

        action = random.randint(0, 1)
        obs, rewards, dones, info = env.step(action)
        print(obs)

        if dones:
            obs = env.reset()
        env.render()
        time.sleep(0.2)
    env.close()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
def main(config, gpu, docker, debug):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu

    experiment(variant)

if __name__ == "__main__":
    main()
