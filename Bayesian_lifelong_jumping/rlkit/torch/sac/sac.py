from collections import OrderedDict
import numpy as np
import copy
import time
import torch
import torch.optim as optim
from torch import nn as nn
from rlkit.core import logger
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm
from rl_alg import BNNdynamics

class BayesianLifelongRL(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            nets,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            encoder_tau=0.005,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.repre_criterion = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.kl_lambda = kl_lambda
        self.encoder_tau = encoder_tau
        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards

        self.forw_dyna_set = nets[1]
        #self.target_vf = self.vf.copy()

        # self.policy_optimizer = optimizer_class(
        #     self.agent.policy.parameters(),
        #     lr=policy_lr,
        # )
        # self.qf1_optimizer = optimizer_class(
        #     self.qf1.parameters(),
        #     lr=qf_lr,
        # )
        # self.qf2_optimizer = optimizer_class(
        #     self.qf2.parameters(),
        #     lr=qf_lr,
        # )
        # self.vf_optimizer = optimizer_class(
        #     self.vf.parameters(),
        #     lr=vf_lr,
        # )
        # self.forward_optimizer_set = []
        # for i in range(len(env)):
        #     self.forward_optimizer = optimizer_class(
        #         self.forw_dyna_set[i].parameters(),
        #         lr=context_lr,
        #     )
        #     self.forward_optimizer_set.append(self.forward_optimizer)


    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent]
# agent.networks: [self.context_encoder, self.policy]
    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def sample_data(self, indices, task=False):
        ''' sample data from replay buffers to construct a training meta-batch '''
        # collect data from multiple tasks for the meta-batch
        obs, actions, rewards, next_obs, terms = [], [], [], [], []
        for idx in indices:
            if task:
                batch = ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size_task))
            else:
                batch = ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size))
            o = batch['observations'][None, ...]
            a = batch['actions'][None, ...]
            r = batch['rewards'][None, ...]
            no = batch['next_observations'][None, ...]
            t = batch['terminals'][None, ...]
            obs.append(o)
            actions.append(a)
            rewards.append(r)
            next_obs.append(no)
            terms.append(t)
        obs = torch.cat(obs, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        next_obs = torch.cat(next_obs, dim=0)
        terms = torch.cat(terms, dim=0)
        return [obs, actions, rewards, next_obs, terms]


    ##### Training #####
    def pretrain(self, env_id, backward=False):
        if backward:
            env_idx = env_id - 50
        else:
            env_idx = env_id

        # obs_dim = self.env_set[0].spec.observation_dim
        # action_dim = self.env_set[0].spec.action_dim
        # self.agent.forw_dyna_set[env_idx] = BNNdynamics(obs_dim, action_dim, device=ptu.device, deterministic=False,weight_out=0.1)
        self.agent.dyna._dynamics_model.set_params(self.agent.dyna._params_mu, self.agent.dyna._params_rho)
        self.agent.forw_dyna_set[env_idx]._params_mu.data.copy_(self.agent.dyna._params_mu.data)
        self.agent.forw_dyna_set[env_idx]._params_rho.data.copy_(self.agent.dyna._params_rho.data)

        self.agent.forw_dyna_set[env_idx]._dynamics_model.set_params(self.agent.forw_dyna_set[env_idx]._params_mu, self.agent.forw_dyna_set[env_idx]._params_rho)
    def _do_training(self, env_idx, backward=False):

        # if self.new_task:
        #     self.agent.dyna.save_old_para()

        if self.task_step // self.global_update_interval != self.update_global:
            logger.push_prefix('Iteration #%d | ' % self.task_step)

            for j in range(self.num_updates_task):
                self._take_step_task(env_idx)
        else:
            for j in range(self.num_updates_task):
                self._take_step_task(env_idx)


    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _take_step_task(self, env_idx):

        obs, actions, rewards, next_obs, terms = self.sample_data([env_idx], task=True)
        t, b, nok = obs.size()

        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        rewards_flat = rewards.view(t * b, -1)
        rewards_flat = rewards_flat * self.reward_scale
        next_obs = next_obs - obs
        elbo, pred, kl, obs_loss, r_loss = self.agent.forw_dyna_set[env_idx].update_posterior(obs, actions, next_obs, rewards_flat, update_post=False, weight_kl=0.0001)
        self.update_step += 1
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            self.eval_statistics['ELBO_task'] = np.mean(elbo)
            self.eval_statistics['loglikelihood_task'] = np.mean(ptu.get_numpy(pred))
            self.eval_statistics['KL_loss_task'] = np.mean(ptu.get_numpy(kl))
            self.eval_statistics['Pred_obs_loss_task'] = np.mean(ptu.get_numpy(obs_loss))
            self.eval_statistics['Pred_rew_loss_task'] = np.mean(ptu.get_numpy(r_loss))
    def _take_step(self, env_id, backward=False):
        if backward:
            self.eval_statistics['Pred_obs_loss'] = 0
            self.eval_statistics['Pred_rew_loss'] = 0
            self.eval_statistics['ELBO'] = 0
            self.eval_statistics['loglikelihood'] = 0
            self.eval_statistics['KL_loss'] = 0
            return
        indices = np.random.choice(env_id + 1, np.min([env_id + 1, self.meta_batch]))
        num_tasks = len(indices)
        obs, actions, rewards, next_obs, terms = self.sample_data(indices)
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        rewards_flat = rewards.view(t * b, -1)
        rewards_flat = rewards_flat * self.reward_scale
        next_obs = next_obs - obs
        elbo, pred, kl, obs_loss, r_loss = self.agent.dyna.update_posterior(obs, actions, next_obs, rewards_flat, update_post=False, weight_kl=0.0001)
        self.global_update_step += 1
        #TODO 0003 Now we are sampling from the whole replay buffer, should we add importance weights to the recent collected transitions?
        #self.curl_optimizer.zero_grad()
        '''self.encoder_optimizer.zero_grad()

        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda / 1000 * kl_div
            kl_loss.backward(retain_graph=True)

        #self.curl_optimizer.step()
        self.encoder_optimizer.step()


        # data is (task, batch, feat)

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension


        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z.detach())
        q2_pred = self.qf2(obs, actions, task_z.detach())
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic



        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z)

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()'''

        # save some statistics for eval
        #if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.


        self.eval_statistics['Pred_obs_loss'] = np.mean(ptu.get_numpy(obs_loss))
        self.eval_statistics['Pred_rew_loss'] = np.mean(ptu.get_numpy(r_loss))
        self.eval_statistics['ELBO'] = np.mean(elbo)
        self.eval_statistics['loglikelihood'] = np.mean(ptu.get_numpy(pred))
        self.eval_statistics['KL_loss'] = np.mean(ptu.get_numpy(kl))
    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            forwardpred=self.agent.dyna.state_dict(),
        )
        return snapshot
