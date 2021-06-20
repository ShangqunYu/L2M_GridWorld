"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch

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
import pickle
import gym

def experiment(variant):


    # env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    # tasks = env.get_all_task_idx()

    SEED = 50  # initial value, 10 will be added for every iteration
    job_name_mtl = 'results/halfcheetah_mtl_bodyparts_exp'
    #job_name_lpgftw = 'results/halfcheetah_lpgftw_gravity_exp'
    torch.set_num_threads(5)
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])

    num_tasks = 60


    f = open(job_name_mtl + '/env_factors.pickle', 'rb')
    size_factors_list = pickle.load(f)
    f.close()
    f = open(job_name_mtl + '/env_ids.pickle', 'rb')
    env_ids = pickle.load(f)
    f.close()
    e_unshuffled = {}
    for task_id in range(num_tasks):
        size_factors = size_factors_list[task_id]
        env_id = env_ids[task_id]
        gym.envs.register(
            id=env_id,
            entry_point='gym_extensions.continuous.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
            max_episode_steps=1000,
            reward_threshold=3800.0,
            kwargs=dict(body_parts=['torso', 'fthigh', 'fshin', 'ffoot'], size_scales=size_factors)
        )
        e_unshuffled[task_id] = GymEnv(env_id)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env = e_unshuffled
    obs_dim = env[0].spec.observation_dim
    action_dim = env[0].spec.action_dim
    print("a:",action_dim)	

    #TODO 0002 Net_size and hyperparameters options


    dyna = BNNdynamics(obs_dim, action_dim, device = ptu.device,learning_rate=0.001,weight_out=0.1)
    qf1_set = []
    qf2_set = []
    vf_set = []
    policy_set = []
    agent_set = []
    forward_dyna_set = []
    for i in range(len(env)):
        # qf1 = FlattenMlp(
        #     hidden_sizes=[net_size, net_size],
        #     input_size=obs_dim + action_dim,
        #     output_size=1,
        # )#qnetwork1
        # qf2 = FlattenMlp(
        #     hidden_sizes=[net_size, net_size],
        #     input_size=obs_dim + action_dim,
        #     output_size=1,
        # )#qnetwork2
        # vf = FlattenMlp(
        #     hidden_sizes=[net_size, net_size],
        #     input_size=obs_dim,
        #     output_size=1,
        # )#qnetwork3?
        # policy = TanhGaussianPolicy(
        #     hidden_sizes=[net_size, net_size],
        #     obs_dim=obs_dim,
        #     action_dim=action_dim,
        # )#actornetwork
        # qf1_set.append(qf1)
        # qf2_set.append(qf2)
        # vf_set.append(vf)
        # policy_set.append(policy)
        forward_dyna = BNNdynamics(obs_dim, action_dim, device=ptu.device, deterministic=False,weight_out=0.1)
        forward_dyna_set.append(forward_dyna)
    agent = Agent(
        forward_dyna=forward_dyna_set,
        dyna=dyna,
        action_dim=action_dim,
        **variant['algo_params']
    )

    algorithm = BayesianLifelongRL(
        env=env,
        nets=[agent, forward_dyna_set],
        **variant['algo_params']
    )

    # optionally load pre-trained weights
    # if variant['path_to_weights'] is not None:
    #     path = variant['path_to_weights']
    #     qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
    #     qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
    #     vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
    #     # TODO hacky, revisit after model refactor
    #     algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
    #     policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    # optional GPU mode

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

