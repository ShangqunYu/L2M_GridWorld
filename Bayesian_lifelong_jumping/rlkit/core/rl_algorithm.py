import abc
from collections import OrderedDict
import time
import copy
import gtimer as gt
import numpy as np

from rlkit.core import logger, eval_util
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.torch import pytorch_util as ptu


class MetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            meta_batch=8,
            num_iterations=100,
            num_pretrain_steps_per_itr=10000,
            num_train_steps_per_itr=500,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=64,
            batch_size_task = 256,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=100,
            discount=0.99,
            replay_buffer_size=300000,
            reward_scale=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            num_updates_global=1500,
            num_updates_task=500,  #change to 500 first
            global_update_interval=300,
            before_train=5000,
            max_traj=10,
            eval_deterministic=True,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.env_set = env
        self.agent = agent
        self.exploration_agent_set = agent # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_pretrain_steps_per_itr = num_pretrain_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.batch_size_task = batch_size_task
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment
        self.num_updates_global = num_updates_global
        self.num_updates_task = num_updates_task
        self.global_update_interval = global_update_interval
        self.before_train = before_train
        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter

        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                len(env),
            )

        self.max_traj = max_traj
        self._n_env_steps_total = 0
        self._n_pretrain_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []

    def make_exploration_policy(self, policy):
         return policy

    def make_eval_policy(self, policy):
        return policy

    def sample_task(self, is_eval=False):
        '''
        sample task randomly
        '''
        if is_eval:
            idx = np.random.randint(len(self.eval_tasks))
        else:
            idx = np.random.randint(len(self.train_tasks))
        return idx

    def train(self):
        '''
        meta-training loop
        '''

        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()
        self.global_update_step = 0
        self.start_train = False
        self.task_step = 0
        self.new_task = False
        backward=False
        self.update_global = -1
        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for env_id in range(len(self.env_set)):
            #env = self.env_set[env_id]
            self.update_step = 0
            if env_id==50:
                backward=True
                # for buff_id in range(30):
                #     self.replay_buffer.task_buffers[buff_id].clear()
            for it_ in gt.timed_for(
                    range(self.num_iterations),
                    save_itrs=True,
            ):

                self._start_epoch(it_ + env_id*self.num_iterations)
                self.training_mode(True)

                if it_ == 0:
                    #self.pretrain(env_id, backward)
                    print('done for pretraining')

                if it_ == 0:
                    self.new_task = True
                else:
                    self.new_task = False
                # Sample train tasks and compute gradient updates on parameters.
                self.try_to_train(env_id, backward)
                # for train_step in range(self.num_train_steps_per_itr):
                #     indices = np.random.choice(self.train_tasks, self.meta_batch)
                #     self._do_training(indices)
                #     self._n_train_steps_total += 1
                gt.stamp('train')

                self.training_mode(False)

                # eval
                if self.start_train and (self.task_step // self.global_update_interval != self.update_global):
                #if self.start_train:
                    self._try_to_eval(it_ + env_id*self.num_iterations)
                    gt.stamp('eval')

                    self._end_epoch()
                    step_temp = self.task_step // self.global_update_interval
                    self.update_global = copy.deepcopy(step_temp)
    def pretrain(self, env_id, backward=False):
        """
        Do anything before the main training phase.
        """
        pass

    def try_to_train(self, env_id, backward=False):
        if backward:
            env_idx = env_id - 50
        else:
            env_idx = env_id

        paths, n_samples, info = self.sampler.obtain_samples(env_idx, max_samples=self.max_path_length,
                                                             max_trajs=self.max_traj, planning=self.start_train)

        self.task_step += n_samples

        if not backward:
            self.replay_buffer.add_paths(env_idx, paths)

        if self.task_step >= self.before_train:

            self.start_train = True
            self._do_training(env_idx, backward)
            print('hello')

            if self.task_step // self.global_update_interval != self.update_global:
                all_rets = []
                online_returns = []
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
                n = min([len(a) for a in all_rets])
                all_rets = [a[:n] for a in all_rets]
                all_rets = np.mean(np.stack(all_rets), axis=0)  # avg return per nth rollout
                online_returns.append(all_rets)

                avg_train_online_return = np.mean(np.stack(online_returns))
                self.eval_statistics['AverageReturn'] = avg_train_online_return
                logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(self.task_step))





    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True, print_success=False):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.agent.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples, info = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                                max_trajs=update_posterior_rate,
                                                                accum_context=False,
                                                                resample=resample_z_rate)
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.prepare_context(self.task_idx)
                self.agent.infer_posterior(context)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def _try_to_eval(self, epoch):
        # logger.save_extra_data(self.get_extra_data_to_save(epoch))

        params = self.get_epoch_snapshot(epoch)
        logger.save_itr_params(epoch, params)
        table_keys = logger.get_table_key_set()
        if self._old_table_keys is not None:
            assert table_keys == self._old_table_keys, (
                "Table keys cannot change from iteration to iteration."
            )
        self._old_table_keys = table_keys

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)

        logger.record_tabular("Epoch", epoch)
        logger.record_tabular("SampleStep", self.task_step)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

        self.eval_statistics = None
    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        #logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self.start_train))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, idx, epoch, run):
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        while num_transitions < self.num_steps_per_eval:
            path, num, info = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1, accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            if num_trajs >= self.num_exp_traj_eval:
                self.agent.infer_posterior(self.agent.context)

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def _do_eval(self, indices, epoch):
        final_returns = []
        online_returns = []
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean([a[-1] for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        if self.dump_eval_paths:
            # 100 arbitrarily chosen for visualizations of point_robot trajectories
            # just want stochasticity of z, not the policy
            self.agent.clear_z()
            prior_paths, _, info1 = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length * 20,
                                                        accum_context=False,
                                                        resample=1)
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        train_suc = []
        for idx in indices:
            self.task_idx = idx
            self.env.reset_task(idx)
            paths = []
            a_s = 0
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                context = self.prepare_context(idx)
                self.agent.infer_posterior(context)
                p, _, info = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length,
                                                        accum_context=False,
                                                        max_trajs=1,
                                                        resample=np.inf)
                a_s += info['n_success_num']
                paths += p
            a_s = a_s / (self.num_steps_per_eval // self.max_path_length)
            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards
            #print(paths)

            train_returns.append(eval_util.get_average_returns(paths))
            train_suc.append(a_s)
            #print(train_returns)
        train_returns = np.mean(train_returns)
        train_suc = np.mean(train_suc)
        #print(train_returns)
        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns = self._do_eval(indices, epoch)
        eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)

        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch)
        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)

        # save the final posterior
        self.agent.log_diagnostics(self.eval_statistics)

        #if hasattr(self.env, "log_diagnostics"):
        #    self.env.log_diagnostics(paths, prefix=None)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        self.eval_statistics['Averagesuc_rate'] = train_suc
        logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass
