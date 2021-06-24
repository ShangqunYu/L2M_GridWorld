import numpy as np
import time
import torch
from torch import nn as nn
import torch.nn.functional as F
from rlkit.torch.core import np_ify
import rlkit.torch.pytorch_util as ptu
from optimizers import CEMOptimizer

def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class Agent(nn.Module):#context encoder -> action output (during training and sampling)
#change plan_hor to 20 from 1 Simon?
    def __init__(self,
                 forward_dyna,
                 dyna,
                 action_dim,
                 per=1,
                 plan_hor=20,
                 npart=20,
                 **kwargs
    ):
        super().__init__()
        self.forw_dyna_set = forward_dyna
        self.dyna = dyna
        self.dU = action_dim
        self.per = per
        self.plan_hor = plan_hor
        self.npart = npart
        self.optimizer = CEMOptimizer(
            sol_dim=self.plan_hor * self.dU,
            lower_bound=np.tile(-np.ones(self.dU), [self.plan_hor]),
            upper_bound=np.tile(np.ones(self.dU), [self.plan_hor]),
            cost_function=self._compile_cost,
            popsize=500,
            num_elites=50,
            max_iters=5,
            alpha=0.1
        )

        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile(np.zeros(self.dU), [self.plan_hor])
        self.init_var = np.tile(np.ones(self.dU) * 1.5, [self.plan_hor])


    def get_action(self, obs, env_idx, get_pred_cost=False, planning=True):
        """Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation
            t: The current timestep
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.

        Returns: An action (and possibly the predicted cost)
        """

        self.current_id = env_idx
        if not planning:
            #simon change to 1 hot action chocie
            return np.eye(2)[np.random.choice(2,1)].squeeze()
            #return np.random.uniform(-1, 1, self.dU)

        if self.ac_buf.shape[0] > 0:

            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]

            return action

        self.sy_cur_obs = obs
        #t1 =time.time()
        #breakpoint()
        soln = self.optimizer.obtain_solution(self.prev_sol, self.init_var)
        #t2 = time.time()
        #print("act:", t2-t1)
        self.prev_sol = np.concatenate([np.copy(soln)[self.per * self.dU:], np.zeros(self.per * self.dU)])
        self.ac_buf = soln[:self.per * self.dU].reshape(-1, self.dU)

        return self.get_action(obs, env_idx)

    @torch.no_grad()
    def _compile_cost(self, ac_seqs):
        #breakpoint()
        nopt = ac_seqs.shape[0]

        ac_seqs = torch.from_numpy(ac_seqs).float().to(ptu.device)

        # Reshape ac_seqs so that it's amenable to parallel compute
        # Before, ac seqs has dimension (400, 25) which are pop size and sol dim coming from CEM
        ac_seqs = ac_seqs.view(-1, self.plan_hor, self.dU)
        #  After, ac seqs has dimension (400, 25, 1)

        transposed = ac_seqs.transpose(0, 1)
        # Then, (25, 400, 1)

        expanded = transposed[:, :, None]
        # Then, (25, 400, 1, 1)

        tiled = expanded.expand(-1, -1, self.npart, -1)
        # Then, (25, 400, 20, 1)

        ac_seqs = tiled.contiguous().view(self.plan_hor, -1, self.dU)
        # Then, (25, 8000, 1)

        # Expand current observation
        cur_obs = torch.from_numpy(self.sy_cur_obs).float().to(ptu.device)
        cur_obs = cur_obs[None]

        cur_obs = cur_obs.expand(nopt * self.npart, -1)

        rews = torch.zeros(nopt, self.npart, device=ptu.device)

        for t in range(self.plan_hor):
            cur_acs = ac_seqs[t]
            proc_obs = cur_obs
            acs = cur_acs
            # proc_obs = self._expand_to_ts_format(cur_obs)
            # acs = self._expand_to_ts_format(cur_acs)

            rew, next_obs = self.forw_dyna_set[self.current_id].infer(proc_obs, acs)


            next_obs = proc_obs + next_obs

            # rew = self._flatten_to_matrix(rew)
            # next_obs = self._flatten_to_matrix(next_obs)
            #

            rew = rew.view(-1, self.npart)

            rews += rew
            cur_obs = next_obs

        # Replace nan with high cost
        rews[rews != rews] = 1e6


        return rews.mean(dim=1).detach().cpu().numpy()

    def _expand_to_ts_format(self, mat):
        dim = mat.shape[-1]

        # Before, [10, 5] in case of proc_obs
        reshaped = mat.view(-1, self.npart, 1, dim)
        # After, [2, 5, 1, 5]

        transposed = reshaped.transpose(0, 1)
        # After, [5, 2, 1, 5]

        reshaped = transposed.contiguous().view(-1, dim)
        # After. [5, 2, 5]

        return reshaped

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.shape[-1]

        reshaped = ts_fmt_arr.view(self.npart, -1, 1, dim)

        transposed = reshaped.transpose(0, 1)

        reshaped = transposed.contiguous().view(-1, dim)

        return reshaped



    def forward(self, obs):

        return obs

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        pass
        #TODO LOG_STATS
        # z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        # z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        # eval_statistics['Z mean eval'] = z_mean
        # eval_statistics['Z variance eval'] = z_sig

    @property
    def networks(self):
        return self.forw_dyna_set + [self.dyna]
