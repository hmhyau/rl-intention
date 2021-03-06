import numpy as np

from base_class import TabularRLModel
from schedules import LinearSchedule, ExponentialSchedule
from gym.spaces import Tuple, Discrete

import cloudpickle as pickle

remove = ['hvalues', 'qvalues', 'policy']

class QTabularRLModel(TabularRLModel):
    def __init__(
        self,
        policy,
        env,
        gamma=0.99,
        learning_rate=1e-2,
        buffer_size=None,
        exploration_type='linear',
        exploration_frac=None, 
        exploration_ep=250,
        exploration_initial_eps=1.,
        exploration_final_eps=0.05, 
        double_q=False,
        policy_kwargs=None,
        seed=None,
        intent=False
        ):
                  
        super(QTabularRLModel, self).__init__(
            policy,
            env,
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
            intent
            )
        
        self._aliases()

    def _aliases(self):
        self.qvalues = self.policy.qvalues
        if self.policy.intent:
            self.hvalues = self.policy.hvalues

    def learn(self, total_timesteps=None, total_episodes=None, log_interval=100, ckpt_interval=100, ckpt_path=None):
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

        train = True
        done = False
        step = 0
        ep_reward = 0
        obs = self.env.reset()

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

            # print(step, next_obs, self.qvalues[next_obs])
            # argmax_a = np.argmax(self.qvalues[next_obs])
            # argmax_a, _ = self.policy.predict(obs, deterministic=True)
            argmax_a = np.argmax(self.qvalues[next_obs])

            if isinstance(self.observation_space, Tuple):
                # print(obs, action)
                expected_reward = reward + self.gamma*self.qvalues[next_obs + (argmax_a,)]*(1-int(done))-self.qvalues[obs + (action,)]
                self.qvalues[obs + (action,)] += self.learning_rate * expected_reward

                if self.policy.intent:
                    intent_update = np.zeros(self.qvalues.shape)
                    intent_update[obs + (action,)] += 1
                    expected_intent = intent_update + self.gamma * self.hvalues[next_obs + (argmax_a,)] * (1-int(done)) - self.hvalues[obs + (action,)]
                    self.hvalues[obs + (action,)] = self.hvalues[obs + (action,)] + self.learning_rate * expected_intent

            if isinstance(self.observation_space, Discrete):
                expected_reward = reward + self.gamma*np.max(self.qvalues[next_obs])*(1-int(done))-self.qvalues[obs, action]
                self.qvalues[obs, action] += self.learning_rate * expected_reward

                if self.policy.intent:
                    intent_update = np.zeros(self.qvalues.shape)
                    intent_update[obs, action] += 1
                    expected_intent = intent_update + self.gamma * self.hvalues[next_obs, argmax_a] * (1-int(done)) - self.hvalues[obs, action]
                    self.hvalues[obs, action] = self.hvalues[obs, action] + self.learning_rate * expected_intent

            obs = next_obs
            step += 1
            ep_reward += reward
            self.elapsed_steps += 1

            if loop_type == 'timesteps':
                if self.elapsed_steps == total_timesteps:
                    train = False

            if done:
                # print(step)
                last_100rewards[self.ep_done%100] = ep_reward
                print("\rEpisode {}/{}, Average Reward {}".format(self.ep_done,total_episodes, np.nanmean(last_100rewards)),end="")
                self.ep_done += 1
                step = 0
                ep_reward = 0
                obs = self.env.reset()
                if loop_type == 'episode':
                    if self.ep_done >= total_episodes:
                        train = False

            if ckpt_path is not None and ckpt_interval:
                if loop_type == 'episode':
                    if self.ep_done % ckpt_interval == 0 and done:
                        ckpt_str = str(self.ep_done)
                        full_path = ckpt_path + '/' + ckpt_str
                        # super(DBNModel, self).save(full_path)
                        super(QTabularRLModel, self).save(full_path)

                if loop_type == 'timesteps':
                    if self.elapsed_steps % ckpt_interval == 0 and done:
                        ckpt_str = str(self.ep_done)
                        full_path = ckpt_path + '/' + ckpt_str
                        # super(DBNModel, self).save(full_path)
                        super(QTabularRLModel, self).save(full_path)


class MCTabularRLModel(TabularRLModel):
    def __init__(
        self,
        policy,
        env,
        gamma=0.99,
        learning_rate=1e-2,
        buffer_size=None, 
        exploration_type='linear',
        exploration_frac=None, 
        exploration_ep=250,
        exploration_initial_eps=1.,
        exploration_final_eps=0.05, 
        double_q=False,
        policy_kwargs=None,
        seed=None,
        intent=False
        ):

        super(MCTabularRLModel, self).__init__(
            policy,
            env,
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
            intent
            )
    
        self._aliases()

    def _aliases(self):
        self.qvalues = self.policy.qvalues
        if self.policy.intent:
            self.hvalues = self.policy.hvalues

    def learn(self, total_timesteps=None, total_episodes=None, log_interval=100, ckpt_interval=100, ckpt_path=None):

        def _sample_episode():
            sample = []
            obs = self.env.reset()
            done = False

            while not done:
                update_eps = self.exploration.value(self.ep_done)
                
                if np.random.random_sample() > update_eps:
                    action, value = self.policy.predict(obs, deterministic=True)
                else:
                    action, value = self.policy.predict(obs, deterministic=False)

                new_obs, reward, done, info = self.env.step(action)

                sample.append((obs, action, reward))
                obs = new_obs

            return sample

        episode_rewards = []
        episode_successes = []
        loop_var = total_timesteps if total_timesteps is not None else total_episodes

        if total_timesteps is not None:
            raise ValueError('Only total_episodes can be specified for this class')

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

        train = True
        ep_reward = 0

        while train:
            sample = _sample_episode()
            obses, actions, rewards = zip(*sample)
            self.ep_reward = np.sum(rewards)
            for idx in range(len(sample)):
                self.elapsed_steps += 1
                discounts = np.array([self.gamma**i for i in range(len(obses)+1)])
                expected_reward = sum(rewards[idx:]*discounts[:-(1+idx)]) - self.qvalues[obses[idx], actions[idx]]
                self.qvalues[obses[idx], actions[idx]] += self.learning_rate * expected_reward
                # print(np.where(self.qvalues!=0))

                if self.policy.intent:
                    intent_update = np.zeros(self.qvalues.shape)
                    for obs, action in zip(obses[idx:], actions[idx:]):
                        intent_update[obs, action] += self.learning_rate
                    tmp = self.hvalues[obses[idx], actions[idx]] * (1-self.learning_rate)
                    tmp += intent_update
                    self.hvalues[obses[idx], actions[idx]] = tmp
                

            self.ep_done += 1
            last_100rewards[self.ep_done%100] = ep_reward
            print("\rEpisode {}/{}, Average Reward {}".format(self.ep_done,total_episodes, np.nanmean(last_100rewards)),end="")
            # print(len(sample))
            ep_reward = 0

            if self.ep_done >= total_episodes:
                train = False

            if ckpt_path is not None and ckpt_interval:
                if loop_type == 'episode':
                    if self.ep_done % ckpt_interval == 0 and done:
                        ckpt_str = str(self.ep_done)
                        full_path = ckpt_path + '/' + ckpt_str
                        # super(DBNModel, self).save(full_path)
                        super(MCTabularRLModel, self).save(full_path)
                if loop_type == 'timesteps':
                    if self.elapsed_steps % ckpt_interval == 0 and done:
                        ckpt_str = str(self.ep_done)
                        full_path = ckpt_path + '/' + ckpt_str
                        # super(DBNModel, self).save(full_path)
                        super(MCTabularRLModel, self).save(full_path)