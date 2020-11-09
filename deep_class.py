import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from base_class import DeepRLModel
from schedules import LinearSchedule, ExponentialSchedule
from gym.spaces import Tuple, Discrete
from policies import DQNPolicy
from utils import convert_to_onehot, as_list, identity

class DQNModel(DeepRLModel):
    def __init__(
        self,
        policy,
        env,
        transform_func=identity,
        gamma=0.99,
        learning_rate=1e-4,
        buffer_size=10000, 
        exploration_frac=None, 
        exploration_ep=250,
        exploration_initial_eps=1.,
        exploration_final_eps=0.05,
        double_q=False,
        policy_kwargs=None,
        seed=None,
        device=None):

        super(DQNModel, self).__init__(
            policy,
            env,
            transform_func,
            gamma,
            learning_rate, 
            buffer_size,
            exploration_frac,
            exploration_ep,
            exploration_initial_eps,
            exploration_final_eps,
            double_q,
            policy_kwargs,
            seed,
            device
            )

        self._aliases()
        print(self.__dict__)

    def _aliases(self):
        self.qnet = self.policy.qnet
        self.qnet_target = self.policy.qnet_target
        # if self.policy.intent:
        #     self.hnet = self.policy.hnet
        self.optimizer = self.policy.optimizer
    
    def learn(self, total_timesteps=None, total_episodes=None, batch_size=16, log_interval=100, log_path=None, ckpt_interval=100, ckpt_path=None):
        writer = None
        if log_path:
            writer = SummaryWriter(log_dir=log_path)

        last_100rewards = np.zeros(100)
        last_100rewards[:] = np.NaN

        if total_timesteps and total_episodes:
            raise ValueError("Only one of total_timesteps or total_episodes can be specified")

        if ckpt_path is None:
            print('Checkpoint path is not provided, no intermediate models will be saved')
    
        loop_type = 'episode' if total_episodes else 'timesteps'
        loop_var = total_timesteps if total_timesteps is not None else total_episodes

        # if self.exploration_frac is None:
        #     self.exploration = LinearSchedule(frac=self.exploration_ep,
        #                                       initial=self.exploration_initial_eps,
        #                                       final=self.exploration_final_eps)
        # else:
        #     self.exploration = LinearSchedule(frac=self.exploration_frac * loop_var,
        #                                       initial=self.exploration_initial_eps,
        #                                       final=self.exploration_final_eps)

        if self.exploration_type == 'linear':
            self.exploration = LinearSchedule(
                frac=self.exploration_frac * loop_var,
                initial=self.exploration_initial_eps,
                final=self.exploration_final_eps)
        elif self.exploration_type == 'exponential':
            self.exploration = ExponentialSchedule(
                frac=self.exploration_frac,
                initial=self.exploration_initial_eps,
                final=self.exploration_final_eps)

        done = False
        step = 0
        ep_reward = 0
        obs = self.env.reset()
        train = True

        while train:
            if loop_type == 'episode':
                update_eps = self.exploration.value(self.ep_done)
            if loop_type == 'timesteps':
                update_eps = self.exploration.value(self.elapsed_steps)

            if np.random.random_sample() > update_eps:
                action, value = self.policy.predict(obs, deterministic=True)
            else:
                action, value = self.policy.predict(obs, deterministic=False)

            next_obs, reward, done, info = self.env.step(action)
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            
            if len(self.replay_buffer) < batch_size * 10:
                if len(self.replay_buffer) == batch_size * 10 or done:
                    self.env.reset()
                continue

            loss = self.train(batch_size=batch_size)

            step += 1
            self.elapsed_steps += 1
            ep_reward += reward

            if writer:
                writer.add_scalar('train/loss', loss, self.elapsed_steps)
                writer.add_scalar('train/epsilon', self.exploration.value, self.elapsed_steps)

            if loop_type == 'timesteps':
                if self.elapsed_steps == total_timesteps:
                    train = False

            if done:
                print(step)
                if self.ep_done % ckpt_interval == 0:
                    self.qnet_target.load_state_dict(self.qnet.state_dict())
                last_100rewards[self.ep_done%100] = ep_reward
                ep_reward = 0
                self.ep_done += 1
                step = 0
                print("\rEpisode {}/{}, Average Reward {}".format(self.ep_done,total_episodes, np.nanmean(last_100rewards)),end="")
                obs = self.env.reset()
                if loop_type == 'episode':
                    if self.ep_done == total_episodes:
                        train = False

            if ckpt_path is not None and ckpt_interval:
                if loop_type == 'episode':
                    if self.ep_done % ckpt_interval == 0 and done:
                        ckpt_str = str(self.ep_done)
                        full_path = ckpt_path + '/' + ckpt_str
                        super(DQNModel, self).save(full_path)
                if loop_type == 'timesteps':
                    if self.elapsed_steps % ckpt_interval == 0 and done:
                        ckpt_str = str(self.ep_done)
                        full_path = ckpt_path + '/' + ckpt_str
                        super(DQNModel, self).save(full_path)


    def train(self, batch_size):
        replay = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            target_q = self.qnet_target(replay.observation_t1.float())
            target_q, _ = target_q.max(dim=1)
            target_q = target_q.reshape(-1, 1)

        target_q = replay.reward + self.gamma*target_q*(1-replay.done)

        current_q = self.qnet(replay.observation.float())
        current_q = current_q.gather(dim=1, index=replay.action)

        current_q = current_q.float()
        target_q = target_q.float()

        loss = F.smooth_l1_loss(current_q, target_q)

        self.policy.optimizer.zero_grad()
        loss.backward()
        for param in self.qnet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy.optimizer.step()

        return loss

class IntentionModel(DeepRLModel):
    # Can't think of a better way to combine both DQN and DBN better so we work with this atm
    def __init__(
        self,
        policy,
        env,
        transform_func=identity,
        gamma=0.99,
        learning_rate=1e-4,
        buffer_size=10000,
        exploration_type='linear',
        exploration_frac=None, 
        exploration_ep=250,
        exploration_initial_eps=1.,
        exploration_final_eps=0.05,
        double_q=False,
        policy_kwargs=None,
        seed=None,
        device=None
        ):

        super(IntentionModel, self).__init__(
            policy,
            env,
            transform_func,
            gamma,
            learning_rate, 
            buffer_size,
            exploration_type,
            exploration_frac,
            exploration_ep,
            exploration_initial_eps,
            exploration_final_eps,
            double_q,
            policy_kwargs,
            seed,
            device
            )

        self.transform_func = transform_func
        self._aliases()

    def _aliases(self):
        self.qnet = self.policy.qnet
        self.qnet_target = self.policy.qnet_target
        self.hnet = self.policy.hnet
        self.hnet_target = self.policy.hnet_target

    def learn(self, 
              total_timesteps=None,
              total_episodes=None,
              batch_size=16,
              log_interval=100,
              log_path=None,
              ckpt_interval=100,
              ckpt_path=None):

        writer = None
        if log_path:
            writer = SummaryWriter(log_dir=log_path)

        last_100rewards = np.zeros(100)
        last_100rewards[:] = np.NaN

        if total_timesteps and total_episodes:
            raise ValueError("Only one of total_timesteps or total_episodes can be specified")

        if ckpt_path is None:
            print('Checkpoint path is not provided, no intermediate models will be saved')
    
        loop_type = 'episode' if total_episodes else 'timesteps'
        loop_var = total_timesteps if total_timesteps is not None else total_episodes

        # if self.exploration_frac is None:
        #     self.exploration = LinearSchedule(frac=self.exploration_ep,
        #                                       initial=self.exploration_initial_eps,
        #                                       final=self.exploration_final_eps)
        # else:
        #     self.exploration = LinearSchedule(frac=self.exploration_frac * loop_var,
        #                                       initial=self.exploration_initial_eps,
        #                                       final=self.exploration_final_eps)

        if self.exploration_type == 'linear':
            self.exploration = LinearSchedule(
                frac=self.exploration_frac * loop_var,
                initial=self.exploration_initial_eps,
                final=self.exploration_final_eps)
        elif self.exploration_type == 'exponential':
            self.exploration = ExponentialSchedule(
                frac=self.exploration_frac,
                initial=self.exploration_initial_eps,
                final=self.exploration_final_eps)

        done = False
        step = 0
        ep_reward = 0
        obs = self.env.reset()
        train = True

        while train:
            if loop_type == 'episode':
                update_eps = self.exploration.value(self.ep_done)
            if loop_type == 'timesteps':
                update_eps = self.exploration.value(self.elapsed_steps)

            if np.random.random_sample() > update_eps:
                action, value, hvalues = self.policy.predict(obs, deterministic=True)
            else:
                action, value, hvalues = self.policy.predict(obs, deterministic=False)

            next_obs, reward, done, info = self.env.step(action)
            self.replay_buffer.add(obs, action, reward, next_obs, int(done))
            obs = next_obs
            
            if len(self.replay_buffer) < batch_size * 10:
                if len(self.replay_buffer) == batch_size * 10 or done:
                    self.env.reset()
                continue

            loss_dqn, loss_dbn = self.train(batch_size=batch_size)

            ep_reward += reward
            step += 1
            self.elapsed_steps += 1

            if writer:
                writer.add_scalar('train/loss', loss, self.elapsed_steps)
                writer.add_scalar('train/epsilon', self.exploration.value, self.elapsed_steps)

            if loop_type == 'timesteps':
                if self.elapsed_steps == total_timesteps:
                    train = False

            if done:
                print(step)
                if self.ep_done % ckpt_interval == 0:
                    self.qnet_target.load_state_dict(self.qnet.state_dict())
                    self.hnet_target.load_state_dict(self.hnet.state_dict())
                last_100rewards[self.ep_done%100] = ep_reward
                ep_reward = 0
                self.ep_done += 1
                step = 0
                print("\rEpisode {}/{}, Average Reward {}".format(self.ep_done,total_episodes, np.nanmean(last_100rewards)),end="")
                obs = self.env.reset()
                if loop_type == 'episode':
                    if self.ep_done == total_episodes:
                        train = False

            if ckpt_path is not None and ckpt_interval:
                if loop_type == 'episode':
                    if self.ep_done % ckpt_interval == 0 and done:
                        ckpt_str = str(self.ep_done)
                        full_path = ckpt_path + '/' + ckpt_str
                        # super(DBNModel, self).save(full_path)
                        super(IntentionModel, self).save(full_path)
                if loop_type == 'timesteps':
                    if self.elapsed_steps % ckpt_interval == 0 and done:
                        ckpt_str = str(self.ep_done)
                        full_path = ckpt_path + '/' + ckpt_str
                        # super(DBNModel, self).save(full_path)
                        super(IntentionModel, self).save(full_path)


    def train(self, batch_size):
        replay = self.replay_buffer.sample(batch_size)

        # DQN
        with torch.no_grad():
            target_q = self.qnet_target(replay.observation_t1.float())
            target_q, idxes = target_q.max(dim=1)
            target_q = target_q.reshape(-1, 1)

            target_q = replay.reward + self.gamma*target_q*(1-replay.done)

        current_q = self.qnet(replay.observation.float())
        current_q = current_q.gather(dim=1, index=replay.action)

        current_q = current_q.float()
        target_q = target_q.float()

        loss_q = F.smooth_l1_loss(current_q, target_q)

        # DBN
        # Not the best but it is the current workaround for gym.Tuple
        if isinstance(self.observation_space, Tuple):
            update = torch.zeros(batch_size, *list(map(lambda x: x.n, self.observation_space)), self.policy.n_actions).to(self.device)
        else:
            update = torch.zeros(batch_size, self.policy.n_obs_dbn, self.policy.n_actions).to(self.device)

        for idx, (observation, action) in enumerate(zip(replay.observation, replay.action)):
            encoded_state = self.transform_func(observation).long().to(self.device)
            if encoded_state.ndim == 0:
                encoded_state = encoded_state.view(-1)
            # print(encoded_state)
            _indexing = []
            if isinstance(encoded_state, (list, tuple)) or (isinstance(encoded_state, torch.Tensor) and len(encoded_state) >= 1):
                _indexing.append(idx)
                for s in encoded_state:
                    _indexing.append(s)
                _indexing.append(action)
            # breakpoint()
            # print(_indexing, update.shape)
            update[_indexing] += 1
            # update[idx, encoded_state, action] += 1

        # Reshape for convenience
        update = update.view(-1, self.policy.n_obs_dbn, self.policy.n_actions)
        # breakpoint()

        with torch.no_grad():
            target_h = self.hnet_target(replay.observation_t1.float())
            target_h = torch.cat([target_h[idx, x, ...] for idx, x in enumerate(idxes.unsqueeze(-1))])

        target_h = update + self.gamma*target_h*(1-replay.done.unsqueeze(-1))

        current_h = self.hnet(replay.observation.float())
        current_h = torch.cat([current_h[idx, x, ...] for idx, x in enumerate(replay.action)])

        current_h = current_h.float()
        target_h = target_h.float()

        loss_h = F.smooth_l1_loss(current_h, target_h)
        # print(loss_q)

        self.policy.optimizer_q.zero_grad()
        loss_q.backward()
        for param in self.qnet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy.optimizer_q.step()

        self.policy.optimizer_h.zero_grad()
        loss_h.backward()
        for param in self.hnet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy.optimizer_h.step()

        return loss_q, loss_h

class IntentionAblatedModel(DeepRLModel):
    # Can't think of a better way to combine both DQN and DBN better so we work with this atm
    def __init__(
        self,
        policy,
        env,
        transform_func=identity,
        gamma=0.99,
        learning_rate=1e-4,
        buffer_size=10000, 
        exploration_type='linear',
        exploration_frac=None, 
        exploration_ep=250,
        exploration_initial_eps=1.,
        exploration_final_eps=0.05,
        double_q=False,
        policy_kwargs=None,
        seed=None,
        device=None
        ):

        super(IntentionAblatedModel, self).__init__(
            policy,
            env,
            transform_func,
            gamma,
            learning_rate, 
            buffer_size,
            exploration_type,
            exploration_frac,
            exploration_ep,
            exploration_initial_eps,
            exploration_final_eps,
            double_q,
            policy_kwargs,
            seed,
            device
            )

        self.transform_func = transform_func
        self._aliases()

    def _aliases(self):
        self.qnet = self.policy.qnet
        self.qnet_target = self.policy.qnet_target
        self.hnet = self.policy.hnet
        self.hnet_target = self.policy.hnet_target

    def learn(self, 
              total_timesteps=None,
              total_episodes=None,
              batch_size=16,
              log_interval=100,
              log_path=None,
              ckpt_interval=100,
              ckpt_path=None):

        writer = None
        if log_path:
            writer = SummaryWriter(log_dir=log_path)

        last_100rewards = np.zeros(100)
        last_100rewards[:] = np.NaN

        if total_timesteps and total_episodes:
            raise ValueError("Only one of total_timesteps or total_episodes can be specified")

        if ckpt_path is None:
            print('Checkpoint path is not provided, no intermediate models will be saved')
    
        loop_type = 'episode' if total_episodes else 'timesteps'
        loop_var = total_timesteps if total_timesteps is not None else total_episodes

        # if self.exploration_frac is None:
        #     self.exploration = LinearSchedule(frac=self.exploration_ep,
        #                                       initial=self.exploration_initial_eps,
        #                                       final=self.exploration_final_eps)
        # else:
        #     self.exploration = LinearSchedule(frac=self.exploration_frac * loop_var,
        #                                       initial=self.exploration_initial_eps,
        #                                       final=self.exploration_final_eps)

        if self.exploration_type == 'linear':
            self.exploration = LinearSchedule(
                frac=self.exploration_frac * loop_var,
                initial=self.exploration_initial_eps,
                final=self.exploration_final_eps)
        elif self.exploration_type == 'exponential':
            self.exploration = ExponentialSchedule(
                frac=self.exploration_frac,
                initial=self.exploration_initial_eps,
                final=self.exploration_final_eps)

        done = False
        step = 0
        obs = self.env.reset()
        train = True
        ep_reward = 0

        while train:
            if loop_type == 'episode':
                update_eps = self.exploration.value(self.ep_done)
            if loop_type == 'timesteps':
                update_eps = self.exploration.value(self.elapsed_steps)

            if np.random.random_sample() > update_eps:
                action, value, hvalues = self.policy.predict(obs, deterministic=True)
            else:
                action, value, hvalues = self.policy.predict(obs, deterministic=False)

            next_obs, reward, done, info = self.env.step(action)
            self.replay_buffer.add(obs, action, reward, next_obs, int(done))
            obs = next_obs
            
            if len(self.replay_buffer) < batch_size * 10:
                if len(self.replay_buffer) == batch_size * 10 or done:
                    self.env.reset()
                continue

            loss_dqn, loss_dbn = self.train(batch_size=batch_size)

            ep_reward += reward
            step += 1
            self.elapsed_steps += 1

            if writer:
                writer.add_scalar('train/loss', loss, self.elapsed_steps)
                writer.add_scalar('train/epsilon', self.exploration.value, self.elapsed_steps)

            if loop_type == 'timesteps':
                if self.elapsed_steps == total_timesteps:
                    train = False

            if done:
                if self.ep_done % ckpt_interval == 0:
                    self.qnet_target.load_state_dict(self.qnet.state_dict())
                    self.hnet_target.load_state_dict(self.hnet.state_dict())
                last_100rewards[ep_reward%100] = ep_reward
                ep_reward = 0
                self.ep_done += 1
                step = 0
                print("\rEpisode {}/{}, Average Reward {}".format(self.ep_done,total_episodes, np.nanmean(last_100rewards)),end="")
                obs = self.env.reset()
                if loop_type == 'episode':
                    if self.ep_done == total_episodes:
                        train = False

            if ckpt_path is not None and ckpt_interval:
                if loop_type == 'episode':
                    if self.ep_done % ckpt_interval == 0 and done:
                        ckpt_str = str(self.ep_done)
                        full_path = ckpt_path + '/' + ckpt_str
                        # super(DBNModel, self).save(full_path)
                        super(IntentionAblatedModel, self).save(full_path)
                if loop_type == 'timesteps':
                    if self.elapsed_steps % ckpt_interval == 0 and done:
                        ckpt_str = str(self.ep_done)
                        full_path = ckpt_path + '/' + ckpt_str
                        # super(DBNModel, self).save(full_path)
                        super(IntentionAblatedModel, self).save(full_path)

    def train(self, batch_size):
        replay = self.replay_buffer.sample(batch_size)

        # DQN
        with torch.no_grad():
            target_q = self.qnet_target(replay.observation_t1.float())
            target_q, idxes = target_q.max(dim=1)
            target_q = target_q.reshape(-1, 1)

            target_q = replay.reward + self.gamma*target_q*(1-replay.done)

        current_q = self.qnet(replay.observation.float())
        current_q = current_q.gather(dim=1, index=replay.action)

        current_q = current_q.float()
        target_q = target_q.float()

        loss_q = F.smooth_l1_loss(current_q, target_q)

        # DBN
        # update = torch.zeros(batch_size, self.policy.n_obs_dbn, self.policy.n_actions).to(self.device)
        # for idx, (observation, action) in enumerate(zip(replay.observation, replay.action)):
        #     encoded_state = self.transform_func(observation).long().to(self.device)
        #     breakpoint()
        #     update[idx, encoded_state, action] += 1

        if isinstance(self.observation_space, Tuple):
            update = torch.zeros(batch_size, *list(map(lambda x: x.n, self.observation_space)), self.policy.n_actions).to(self.device)
        else:
            update = torch.zeros(batch_size, self.policy.n_obs_dbn, self.policy.n_actions).to(self.device)

        for idx, (observation, action) in enumerate(zip(replay.observation, replay.action)):
            encoded_state = self.transform_func(observation).long().to(self.device)
            if encoded_state.ndim == 0:
                encoded_state = encoded_state.view(-1)
            # print(encoded_state)
            _indexing = []
            if isinstance(encoded_state, (list, tuple)) or (isinstance(encoded_state, torch.Tensor) and len(encoded_state) >= 1):
                _indexing.append(idx)
                for s in encoded_state:
                    _indexing.append(s)
                _indexing.append(action)
            # breakpoint()
            update[_indexing] += 1
            # update[idx, encoded_state, action] += 1

        # Reshape for convenience
        update = update.view(-1, self.policy.n_obs_dbn, self.policy.n_actions)
        # breakpoint()

        with torch.no_grad():
            target_h = self.hnet_target(replay.observation_t1, idxes.unsqueeze(-1))
            # target_h = torch.cat([target_h[idx, x, ...] for idx, x in enumerate(idxes.unsqueeze(-1))])

        target_h = update + self.gamma*target_h*(1-replay.done.unsqueeze(-1))

        current_h = self.hnet(replay.observation, replay.action)
        # current_h = torch.cat([current_h[idx, x, ...] for idx, x in enumerate(replay.action)])
        # current_h = current_h.gather(dim=1, index=replay.action)

        current_h = current_h.float()
        target_h = target_h.float()

        loss_h = F.smooth_l1_loss(current_h, target_h)
        # print(loss_h)

        self.policy.optimizer_q.zero_grad()
        loss_q.backward()
        for param in self.qnet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy.optimizer_q.step()

        self.policy.optimizer_h.zero_grad()
        loss_h.backward()
        for param in self.hnet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy.optimizer_h.step()

        return loss_q, loss_h