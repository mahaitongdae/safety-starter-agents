#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: buffer.py
# =====================================

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from safe_rl.utils.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, args, buffer_id):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
          Max number of transitions to store in the buffer. When the buffer
          overflows the old memories are dropped.
        """
        self.args = args
        self.buffer_id = buffer_id
        self._storage = []
        self._maxsize = self.args.max_buffer_size
        self._next_idx = 0
        self.replay_starts = self.args.replay_starts
        self.replay_batch_size = self.args.replay_batch_size
        self.stats = {}
        self.replay_times = 0
        logger.info('Buffer initialized')

    def get_stats(self):
        self.stats.update(dict(storage=len(self._storage)))
        return self.stats

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, weight):
        data = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), \
               np.array(obses_tp1), np.array(dones)

    def sample_idxes(self, batch_size):
        return np.array([random.randint(0, len(self._storage) - 1) for _ in range(batch_size)], dtype=np.int32)

    def sample_with_idxes(self, idxes):
        return list(self._encode_sample(idxes)) + [idxes,]

    def sample(self, batch_size):
        idxes = self.sample_idxes(batch_size)
        return self.sample_with_idxes(idxes)

    def add_batch(self, batch):
        for trans in batch:
            self.add(*trans, 0)

    def replay(self):
        if len(self._storage) < self.replay_starts:
            return None
        if self.buffer_id == 1 and self.replay_times % self.args.buffer_log_interval == 0:
            logger.info('Buffer info: {}'.format(self.get_stats()))

        self.replay_times += 1
        return self.sample(self.replay_batch_size)

class ReplayBufferWithCost(object):
    def __init__(self, size, buffer_id=0):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
          Max number of transitions to store in the buffer. When the buffer
          overflows the old memories are dropped.
        """
        # self.args = args
        self.buffer_id = buffer_id
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        # self.replay_starts = self.args.replay_starts
        # self.replay_batch_size = self.args.replay_batch_size
        self.stats = {}
        self.replay_times = 0

    def get_stats(self):
        self.stats.update(dict(storage=len(self._storage)))
        return self.stats

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, cost, weight):
        data = (obs_t, action, reward, obs_tp1, done, cost)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, costs = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, cost = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            costs.append(cost)
            # self.batch_data = {'batch_obs': batch_data[0].astype(np.float32),
            #                    'batch_actions': batch_data[1].astype(np.float32),
            #                    'batch_rewards': batch_data[2].astype(np.float32),
            #                    'batch_obs_tp1': batch_data[3].astype(np.float32),
            #                    'batch_dones': batch_data[4].astype(np.float32),
            #                    }
        # return np.array(obses_t), np.array(actions), np.array(rewards), \
        #        np.array(obses_tp1), np.array(dones), np.array(costs)
        return dict(obs1=np.array(obses_t).tolist(),
                    obs2=np.array(obses_tp1).tolist(),
                    acts=np.array(actions).tolist(),
                    rews=np.array(rewards).tolist(),
                    costs=np.array(costs).tolist(),
                    done=np.array(dones).tolist())

    def sample_idxes(self, batch_size):
        return np.array([random.randint(0, len(self._storage) - 1) for _ in range(batch_size)], dtype=np.int32)

    # def sample_with_idxes(self, idxes):
    #     return list(self._encode_sample(idxes)) + [idxes,]

    def sample_batch(self, batch_size):
        idxes = self.sample_idxes(batch_size)
        return self._encode_sample(idxes), idxes

    # def sample(self, batch_size):
    #     idxes = self.sample_idxes(batch_size)
    #     return self.sample_with_idxes(idxes)

    def add_batch(self, batch):
        for trans in batch:
            self.add(*trans, 0)

    def replay(self, batch_size):
        # if len(self._storage) < self.replay_starts:
        #     return None
        # if self.buffer_id == 1 and self.replay_times % self.args.buffer_log_interval == 0:
        #     logger.info('Buffer info: {}'.format(self.get_stats()))
        self.replay_times += 1
        return self.sample(batch_size)


class PrioritizedReplayBuffer(ReplayBufferWithCost):
    def __init__(self, size, buffer_id=1, replay_alpha=0.4, replay_beta=0.6):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
          Max number of transitions to store in the buffer. When the buffer
          overflows the old memories are dropped.
        alpha: float
          how much prioritization is used
          (0 - no prioritization, 1 - full prioritization)
        beta: float
          To what degree to use importance weights
          (0 - no corrections, 1 - full correction)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size, buffer_id)
        assert replay_alpha > 0
        self._alpha = replay_alpha
        self._beta = replay_beta

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t, action, reward, obs_tp1, done, costs, weight):
        """See ReplayBuffer.store_effect"""

        idx = self._next_idx
        super(PrioritizedReplayBuffer, self).add(obs_t, action, reward,
                                                 obs_tp1, done, costs, weight)
        if weight is None:
            weight = self._max_priority
        self._it_sum[idx] = weight ** self._alpha
        self._it_min[idx] = weight ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage))
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return np.array(res, dtype=np.int32)

    def sample_idxes(self, batch_size):
        return self._sample_proportional(batch_size)

    def sample_with_weights_and_idxes(self, idxes):
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-self._beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-self._beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return list(encoded_sample) + [weights, idxes]

    def sample(self, batch_size):
        idxes = self.sample_idxes(batch_size)
        return self.sample_with_weights_and_idxes(idxes)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
          List of idxes of sampled transitions
        priorities: [float]
          List of updated priorities corresponding to
          transitions at the sampled idxes denoted by
          variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority >= 0 #todo: modify > to >=
            assert 0 <= idx < len(self._storage)
            delta = priority ** self._alpha - self._it_sum[idx]
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

