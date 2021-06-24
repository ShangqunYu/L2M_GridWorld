import numpy as np
import time
import torch
import rlkit.torch.pytorch_util as ptu
import os

def save_prediction(data):
    directory = 'logs/'
    timestamp = time.time()
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory+str(timestamp)+".csv"
    #breakpoint()
    #data = np.array(data)
    np.savetxt(filename,data,fmt = '%s')
    #with open(filename, "w") as text_file:
        #text_file.write(str(data))

def rollout(env, agent, env_idx, max_path_length=np.inf, planning=True,accum_context=True, resample_z=False, animated=False, save_frames=False):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :param accum_context: if True, accumulate the collected context
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    r_los = []
    o_los = []
    o = env.reset()
    goal_step = 0
    path_length = 0
    success_num = 0
    #comparing prediction vs reality
    prediction_data = []
    if animated:
        env.render()

    while path_length < max_path_length:
        #print(o)

        a = agent.get_action(o, env_idx, planning=planning)

        #estimated reward and next state given by our model
        #note! r_ is not next state, it is the different between the current state and next state

        r_, s_, = agent.forw_dyna_set[env_idx].infer(torch.from_numpy(o).float().to(ptu.device)[None],torch.from_numpy(a).float().to(ptu.device)[None])

        next_o, r, d, env_info = env.step(a)
        #print("obs:", next_o)
        #print("vel:", env.env.env.sim.data.qvel.flat)
        # print("pos:", env.env.env.sim.data.qpos.flat[1:])
        pred_no = s_.detach().cpu().numpy() + o
        pred_r = r_.detach().cpu().numpy()
        #print("pred_obs", pred_no)
        #store predicted obs and reward with real ones.
        temp_data_prediction = [next_o.tolist(), pred_no.tolist(), r, pred_r.tolist()]
        prediction_data.append(temp_data_prediction)
        #next_o[-9:] = next_o[-9:] / 5
        # o_los.append(np.sum((pred_no-next_o)**2))
        # r_los.append((pred_r - r) ** 2)
        # update the agent's current context

        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append([1])
        path_length += 1
        goal_step += 1

        if save_frames:
            from PIL import Image
            image = Image.fromarray(np.flipud(env.get_image()))
            env_info['frame'] = image
        env_infos.append(env_info)
        #if env_info['success'] == 1:
        success_num += 1
        #    break
        if goal_step > max_path_length or d:
            break
        o = next_o
        if animated:
            env.render()
    # print("r_pred_loss:", np.mean(r_los))
    # print("no_pred_loss:", np.mean(o_los))
    save_prediction(prediction_data)
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )

    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        success=success_num,
    )


def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs


def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]
