#!/usr/bin/env python

from functools import partial
import numpy as np
import tensorflow as tf
import gym
import time
import os
from safe_rl.utils.logx import EpochLogger
from safe_rl.utils.mpi_tf import sync_all_params, MpiAdamOptimizer
from safe_rl.utils.mpi_tools import mpi_fork, mpi_sum, proc_id, mpi_statistics_scalar, num_procs
from safe_rl.utils.buffer import ReplayBufferWithCost, PrioritizedReplayBuffer

EPS = 1e-8

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, output_bias=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    if output_bias == None:
        return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)
    else:
        return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation,
                               bias_initializer=tf.constant_initializer(output_bias))

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def get_target_update(main_name, target_name, polyak):
    ''' Get a tensorflow op to update target variables based on main variables '''
    main_vars = {x.name: x for x in get_vars(main_name)}
    targ_vars = {x.name: x for x in get_vars(target_name)}
    assign_ops = []
    for v_targ in targ_vars:
        assert v_targ.startswith(target_name), f'bad var name {v_targ} for {target_name}'
        v_main = v_targ.replace(target_name, main_name, 1)
        assert v_main in main_vars, f'missing var name {v_main}'
        assign_op = tf.assign(targ_vars[v_targ], polyak*targ_vars[v_targ] + (1-polyak)*main_vars[v_main])
        assign_ops.append(assign_op)
    return tf.group(assign_ops)


"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)
    log_std = tf.layers.dense(net, act_dim, activation=None)
    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    # Adjustment to log prob
    logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)

    # Squash those unbounded actions!
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi


"""
Actors and Critics
"""
def mlp_actor(x, a, name='pi', hidden_sizes=(256,256), activation=tf.nn.relu,
              output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    # policy
    with tf.variable_scope(name):
        mu, pi, logp_pi = policy(x, a, hidden_sizes, activation, output_activation)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    # make sure actions are in correct range
    action_scale = action_space.high[0]
    mu *= action_scale
    pi *= action_scale

    return mu, pi, logp_pi


def mlp_critic(x, a, pi, name, hidden_sizes=(256,256), activation=tf.nn.elu,
               output_activation=None, policy=mlp_gaussian_policy, action_space=None):

    fn_mlp = lambda x : tf.squeeze(mlp(x=x,
                                       hidden_sizes=list(hidden_sizes)+[1],
                                       activation=activation,
                                       output_activation=output_activation),
                                   axis=1)
    with tf.variable_scope(name):
        critic = fn_mlp(tf.concat([x,a], axis=-1))

    with tf.variable_scope(name, reuse=True):
        critic_pi = fn_mlp(tf.concat([x,pi], axis=-1))

    return critic, critic_pi

def mlp_lam(x, a, pi, name, hidden_sizes=(256,256), activation=tf.nn.relu, output_activation=tf.nn.softplus,
            policy=mlp_gaussian_policy, action_space=None, output_bias=None, lam_obs_only=False):

    fn_mlp = lambda x : tf.squeeze(mlp(x=x,
                                       hidden_sizes=list(hidden_sizes)+[1],
                                       activation=activation,
                                       output_activation=output_activation,
                                       output_bias=output_bias),
                                   axis=1)
    with tf.variable_scope(name):
        if lam_obs_only:
            lam = fn_mlp(tf.concat([x], axis=-1))
        else:
            lam = fn_mlp(tf.concat([x,a], axis=-1))

    # with tf.variable_scope(name, reuse=True):
    #     critic_pi = fn_mlp(tf.concat([x,pi], axis=-1))

    return lam


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.costs_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, cost):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.costs_buf[self.ptr] = cost
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    costs=self.costs_buf[idxs],
                    done=self.done_buf[idxs])


"""
Soft Actor-Critic
"""
def fsac(env_fn, actor_fn=mlp_actor, critic_fn=mlp_critic, lam_fn=mlp_lam, ac_kwargs=dict(), seed=0,
        steps_per_epoch=1000, epochs=100, replay_size=int(1e6), gamma=0.99, cost_gamma=0.99,
        polyak=0.995, lr=1e-4, lam_lr=5e-6, batch_size=256*16, local_start_steps=int(1e3),
        max_ep_len=1000, logger_kwargs=dict(), save_freq=10, local_update_after=int(1e3),
        update_freq=100, render=False,
        fixed_entropy_bonus=None, entropy_constraint=-1.0,
        fixed_cost_penalty=None, cost_constraint=None, cost_lim=None,
        reward_scale=1, pointwise_multiplier=False, dual_ascent_inverval=20, max_lam_grad_norm=10.0,
        prioritized_experience_replay=False, constrained_costs=True, double_qc=True, lam_obs_only=False, **kwargs
        ):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_fn: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the actor
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ===========  ================  ======================================

        critic_fn: A function which takes in placeholder symbols
            for state, ``x_ph``, action, ``a_ph``, and policy ``pi``,
            and returns the critic outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``critic``    (batch,)         | Gives one estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``critic_pi`` (batch,)         | Gives another estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_fn / critic_fn
            function you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        batch_size (int): Minibatch size for SGD.

        local_start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        fixed_entropy_bonus (float or None): Fixed bonus to reward for entropy.
            Units are (points of discounted sum of future reward) / (nats of policy entropy).
            If None, use ``entropy_constraint`` to set bonus value instead.

        entropy_constraint (float): If ``fixed_entropy_bonus`` is None,
            Adjust entropy bonus to maintain at least this much entropy.
            Actual constraint value is multiplied by the dimensions of the action space.
            Units are (nats of policy entropy) / (action dimenson).

        fixed_cost_penalty (float or None): Fixed penalty to reward for cost.
            Units are (points of discounted sum of future reward) / (points of discounted sum of future costs).
            If None, use ``cost_constraint`` to set penalty value instead.

        cost_constraint (float or None): If ``fixed_cost_penalty`` is None,
            Adjust cost penalty to maintain at most this much cost.
            Units are (points of discounted sum of future costs).
            Note: to get an approximate cost_constraint from a cost_lim (undiscounted sum of costs),
            multiply cost_lim by (1 - gamma ** episode_len) / (1 - gamma).
            If None, use cost_lim to calculate constraint.

        cost_lim (float or None): If ``cost_constraint`` is None,
            calculate an approximate constraint cost from this cost limit.
            Units are (expectation of undiscounted sum of costs in a single episode).
            If None, cost_lim is not used, and if no cost constraints are used, do naive optimization.
    """
    # use_costs = fixed_cost_penalty or cost_constraint or cost_lim
    use_costs = constrained_costs

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Env instantiation
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Setting seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    test_env.seed(seed)

    # Create summary writer
    logs_dir = logger_kwargs.get('output_dir') + '/fsac'
    summary_writer = tf.summary.FileWriter(logs_dir, tf.get_default_graph())

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph, c_ph = placeholders(obs_dim, act_dim, obs_dim, None, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi = actor_fn(x_ph, a_ph, **ac_kwargs)
        qr1, qr1_pi = critic_fn(x_ph, a_ph, pi, name='qr1', **ac_kwargs)
        qr2, qr2_pi = critic_fn(x_ph, a_ph, pi, name='qr2', **ac_kwargs)
        qc, qc_pi = critic_fn(x_ph, a_ph, pi, name='qc', output_activation=tf.nn.softplus, **ac_kwargs)
        if double_qc:
            qc2, qc2_pi = critic_fn(x_ph, a_ph, pi, name='qc2', output_activation=tf.nn.softplus, **ac_kwargs)
            tf.summary.histogram('Optimizer/QC2Vals', qc2)

    tf.summary.histogram('Optimizer/QCVals', qc)
    tf.summary.histogram('Optimizer/QR1Vals', qr1)
    tf.summary.histogram('Optimizer/QR2Vals', qr2)


    with tf.variable_scope('main', reuse=True):
        # Additional policy output from a different observation placeholder
        # This lets us do separate optimization updates (actor, critics, etc)
        # in a single tensorflow op.
        _, pi2, logp_pi2 = actor_fn(x2_ph, a_ph, **ac_kwargs)

    # Target value network
    with tf.variable_scope('target'):
        _, qr1_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qr1', **ac_kwargs)
        _, qr2_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qr2', **ac_kwargs)
        _, qc_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qc',  **ac_kwargs)
        if double_qc:
            _, qc2_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qc2', **ac_kwargs)

    # Entropy bonus
    if fixed_entropy_bonus is None:
        with tf.variable_scope('entreg'):
            soft_alpha = tf.get_variable('soft_alpha',
                                         initializer=0.0,
                                         trainable=True,
                                         dtype=tf.float32)
        alpha = tf.nn.softplus(soft_alpha)
    else:
        alpha = tf.constant(fixed_entropy_bonus)
    log_alpha = tf.log(alpha)
    tf.summary.scalar('Optimizer/Alpha', alpha)
    tf.summary.scalar('Optimizer/LogAlpha', log_alpha)

    # Cost penalty
    if use_costs:
        if fixed_cost_penalty is None:
            if pointwise_multiplier is False:
                with tf.variable_scope('costpen'):
                    soft_beta = tf.get_variable('soft_beta',
                                                 initializer=0.0,
                                                 trainable=True,
                                                 dtype=tf.float32)
                beta = tf.nn.softplus(soft_beta)
                log_beta = tf.log(beta)
                tf.summary.scalar('Optimizer/Beta', beta)
                tf.summary.scalar('Optimizer/LogBeta', log_beta)
            else:
                with tf.variable_scope('main', reuse=tf.AUTO_REUSE):
                    lam = lam_fn(x_ph, a_ph, pi, name='lam', lam_obs_only=lam_obs_only, **ac_kwargs)
                    tf.summary.histogram('Optimizer/Lam', lam)

        else:
            beta = tf.constant(fixed_cost_penalty)
            log_beta = tf.log(beta)
    else:
        beta = 0.0  # costs do not contribute to policy optimization
        print('Not using costs')

    # Experience buffer
    # replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    if prioritized_experience_replay:
        replay_buffer = PrioritizedReplayBuffer(replay_size)
    else:
        replay_buffer = ReplayBufferWithCost(replay_size)

    # Count variables
    if proc_id()==0:
        if pointwise_multiplier is False:
            var_counts = tuple(count_vars(scope) for scope in
                               ['main/pi', 'main/qr1', 'main/qr2', 'main/qc', 'main'])
            print(('\nNumber of parameters: \t pi: %d, \t qr1: %d, \t qr2: %d, \t qc: %d, \t  total: %d\n')%var_counts)
        else:
            var_counts = tuple(count_vars(scope) for scope in
                               ['main/pi', 'main/qr1', 'main/qr2', 'main/qc', 'main/lam', 'main'])
            print(
                ('\nNumber of parameters: \t pi: %d, \t qr1: %d, \t qr2: %d, \t qc: %d, \t lam: %d, \t  total: %d\n') % var_counts)

    # Min Double-Q:
    min_q_pi = tf.minimum(qr1_pi, qr2_pi)
    min_q_pi_targ = tf.minimum(qr1_pi_targ, qr2_pi_targ)

    if double_qc:
        max_qc = tf.maximum(qc, qc2)
        max_qc_pi = tf.maximum(qc_pi, qc2_pi)
        max_qc_pi_targ = tf.maximum(qc_pi_targ, qc2_pi_targ)
        qc_backup = tf.stop_gradient(tf.clip_by_value(c_ph + cost_gamma * (1 - d_ph) * qc_pi_targ, 0, 100))
    else:
        qc_backup = tf.stop_gradient(tf.clip_by_value(c_ph + cost_gamma * (1 - d_ph) * qc_pi_targ, 0, 100))

    # Targets for Q and V regression
    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*(min_q_pi_targ - alpha * logp_pi2))
    tf.summary.histogram('Optimizer/QCTarget', qc_backup)
    tf.summary.histogram('Optimizer/QTarget', q_backup)

    # Soft actor-critic losses
    if pointwise_multiplier:
        if double_qc:
            penalty_terms = lam * max_qc_pi
            pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi + penalty_terms)
        else:
            penalty_terms = lam * qc_pi
            pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi + penalty_terms)
        tf.summary.scalar('Optimizer/PenaltyTerms', tf.reduce_mean(penalty_terms))
        tf.summary.scalar('Optimizer/LossPi', pi_loss)
    else:
        if double_qc:
            pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi + beta * max_qc_pi)
        else:
            pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi + beta * qc_pi)
        tf.summary.scalar('Optimizer/LossPi', pi_loss)
    qr1_loss = 0.5 * tf.reduce_mean((q_backup - qr1)**2)
    qr2_loss = 0.5 * tf.reduce_mean((q_backup - qr2)**2)
    qc_td_error = qc_backup - qc
    if double_qc:
        qc1_loss = 0.5 * tf.reduce_mean((qc_backup - qc)**2)
        qc2_loss = 0.5 * tf.reduce_mean((qc_backup - qc2)**2)
        qc_loss = qc1_loss + qc2_loss
        tf.summary.scalar('Optimizer/LossQC1', qc1_loss)
        tf.summary.scalar('Optimizer/LossQC2', qc2_loss)
    else:
        qc_loss = 0.5 * tf.reduce_mean(qc_td_error**2)
    q_loss = qr1_loss + qr2_loss + qc_loss
    tf.summary.scalar('Optimizer/LossQR1', qr1_loss)
    tf.summary.scalar('Optimizer/LossQR2', qr2_loss)
    tf.summary.scalar('Optimizer/LossQC', qc_loss)
    tf.summary.histogram('Optimizer/TDErrorQC', qc_td_error)



    # Loss for alpha
    entropy_constraint *= act_dim
    pi_entropy = -tf.reduce_mean(logp_pi)
    # alpha_loss = - soft_alpha * (entropy_constraint - pi_entropy)
    alpha_loss = - alpha * (entropy_constraint - pi_entropy)
    print('using entropy constraint', entropy_constraint)
    tf.summary.scalar('Optimizer/PiEntrophy',pi_entropy)
    tf.summary.scalar('Optimizer/LossAlpha', alpha_loss)

    if cost_constraint is None:
        # Convert assuming equal cost accumulated each step
        # Note this isn't the case, since the early in episode doesn't usually have cost,
        # but since our algorithm optimizes the discounted infinite horizon from each entry
        # in the replay buffer, we should be approximately correct here.
        # It's worth checking empirical total undiscounted costs to see if they match.
        cost_constraint = cost_lim * (1 - cost_gamma ** max_ep_len) / (1 - cost_gamma) / max_ep_len
    print('using cost constraint', cost_constraint)
    if double_qc:
        violation = max_qc - cost_constraint
        vios_count = tf.where(max_qc > cost_constraint, tf.ones_like(qc), tf.zeros_like(qc))
    else:
        violation = qc - cost_constraint
        vios_count = tf.where(qc > cost_constraint, tf.ones_like(qc), tf.zeros_like(qc))
    vios_rate = tf.reduce_sum(vios_count) / tf.convert_to_tensor(batch_size / 16, dtype=tf.float32) # todo:cpus
    tf.summary.scalar('Optimizer/ViolationRate', vios_rate)
    tf.summary.histogram('Optimizer/Violation', violation)

    # Loss for beta
    if use_costs:
        if pointwise_multiplier is False:
            # beta_loss = beta * (cost_constraint - qc)
            beta_loss = - beta * violation
            tf.summary.scalar('Optimizer/LossBeta', tf.reduce_mean(beta_loss))
        else:
            lam_loss = - tf.reduce_mean(lam * violation) # tf.clip_by_value(violation, -0.1, 100)
            tf.summary.scalar('Optimizer/LossLam', lam_loss)


    local_step = tf.Variable(0.0, trainable=False)
    decayed_lr = tf.train.polynomial_decay(learning_rate=lr,global_step=local_step,decay_steps=epochs*steps_per_epoch,end_learning_rate=8e-6)

    tf.summary.scalar('Optimizer/lr', decayed_lr)
    merged_summary = tf.summary.merge_all()

    # Policy train op
    # (has to be separate from value train op, because qr1_pi appears in pi_loss)
    train_pi_op = MpiAdamOptimizer(learning_rate=decayed_lr).minimize(pi_loss, var_list=get_vars('main/pi'),
                                                                      name='train_pi') # , global_step=local_step

    # Value train op
    with tf.control_dependencies([train_pi_op]):
        train_q_op = MpiAdamOptimizer(learning_rate=decayed_lr).minimize(q_loss, var_list=get_vars('main/q'), name='train_q')

    if fixed_entropy_bonus is None:
        entreg_optimizer = MpiAdamOptimizer(learning_rate=lr)
        with tf.control_dependencies([train_q_op]):
            train_entreg_op = entreg_optimizer.minimize(alpha_loss, var_list=get_vars('entreg'))

    if use_costs and fixed_cost_penalty is None:
        cost_optimizer = MpiAdamOptimizer(learning_rate=lr)
        with tf.control_dependencies([train_entreg_op]):
            if pointwise_multiplier is False:
                train_costpen_op = cost_optimizer.minimize(beta_loss, var_list=get_vars('costpen'))
    if pointwise_multiplier:
        local_ascent_step = tf.Variable(0.0, trainable=False)
        decayed_lam_lr = tf.train.polynomial_decay(learning_rate=lam_lr, global_step=local_ascent_step,
                                                   decay_steps=epochs*steps_per_epoch/dual_ascent_inverval,
                                                   end_learning_rate=2e-6)
        lam_optimizer = MpiAdamOptimizer(learning_rate=decayed_lam_lr)
        grads, vars = zip(*lam_optimizer.compute_gradients(lam_loss, var_list=get_vars('main/lam')))
        grads, grad_norm = tf.clip_by_global_norm(grads, max_lam_grad_norm)
        train_lam_op = lam_optimizer.apply_gradients(list(zip(grads, vars))) # ,global_step=local_ascent_step
        # train_lam_op = lam_optimizer.minimize(lam_loss, var_list=get_vars('main/lam'))
        # tf.summary.scalar('Optimizer/LamGradNorm', grad_norm)


    # Polyak averaging for target variables
    target_update = get_target_update('main', 'target', polyak)

    # Single monolithic update with explicit control dependencies
    with tf.control_dependencies([train_pi_op]):
        with tf.control_dependencies([train_q_op]):
            grouped_update = tf.group([target_update])

    if fixed_entropy_bonus is None:
        grouped_update = tf.group([grouped_update, train_entreg_op])
    if use_costs and fixed_cost_penalty is None:
        if pointwise_multiplier is False:
            grouped_update = tf.group([grouped_update, train_costpen_op])
        # else:
        #     grouped_update = tf.group([grouped_update, train_lam_op])

    # Initializing targets to match main variables
    # As a shortcut, use our exponential moving average update w/ coefficient zero
    target_init = get_target_update('main', 'target', 0.0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    if pointwise_multiplier:
        logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                                    outputs={'mu': mu, 'pi': pi, 'qr1': qr1, 'qr2': qr2, 'qc': qc, 'lam':lam})
    else:
        logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                              outputs={'mu': mu, 'pi': pi, 'qr1': qr1, 'qr2': qr2, 'qc': qc})

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_cost, ep_len, ep_goals, = test_env.reset(), 0, False, 0, 0, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time !!! critical
                a = get_action(o, True)
                o2, r, d, info = test_env.step(a)
                if render and proc_id() == 0 and j == 0:
                    test_env.render()
                ep_ret += r
                c = info.get('cost', 0)
                ep_cost += info.get('cost', 0)
                ep_len += 1
                ep_goals += 1 if info.get('goal_met', False) else 0
                # replay_buffer.add(o, a, r, o2, d, c, None) # add test samples in buffer? actually add deterministic action exploration
                o = o2
            logger.store(TestEpRet=ep_ret, TestEpCost=ep_cost, TestEpLen=ep_len, TestEpGoals=ep_goals)
            test_summary=dict(TestEpRet=ep_ret, TestEpCost=ep_cost, TestEpLen=ep_len, TestEpGoals=ep_goals)
            # summary_writer.add_summary()

    start_time = time.time()
    o, r, d, ep_ret, ep_cost, ep_len, ep_goals = env.reset(), 0, False, 0, 0, 0, 0
    total_steps = steps_per_epoch * epochs # 8000*100 = 8e5

    # variables to measure in an update
    if pointwise_multiplier:
        vars_to_get = dict(LossPi=pi_loss, LossQR1=qr1_loss, LossQR2=qr2_loss, LossQC=qc_loss,
                           QR1Vals=qr1, QR2Vals=qr2, QRTarget=q_backup, QCVals=qc, LogPi=logp_pi, PiEntropy=pi_entropy,
                           Alpha=alpha, LogAlpha=log_alpha, LossAlpha=alpha_loss, QCTarget=qc_backup, QCTDError=qc_td_error,
                           PenaltyTerms=penalty_terms, LossLam=lam_loss, Violation=violation, ViolationRate=vios_rate, ViolationCount=vios_count)
    else:
        vars_to_get = dict(LossPi=pi_loss, LossQR1=qr1_loss, LossQR2=qr2_loss, LossQC=qc_loss,
                           QR1Vals=qr1, QR2Vals=qr2, QRTarget=q_backup, QCVals=qc, LogPi=logp_pi, PiEntropy=pi_entropy,
                           Alpha=alpha, LogAlpha=log_alpha, LossAlpha=alpha_loss, QCTarget=qc_backup, QCTDError=qc_td_error,
                           Violation=violation, ViolationRate=vios_rate)
    if double_qc:
        vars_to_get.update(dict(LossQC1=qc1_loss, LossQC2=qc2_loss,QC2Vals=qc2))
    # , QcLim=cost_constraint not allowed, here vars_to_get is a Tensor
    if use_costs:
        if pointwise_multiplier is False:
            vars_to_get.update(dict(Beta=beta, LogBeta=log_beta, LossBeta=beta_loss))
        else:
            vars_to_get.update(dict(Lam=lam))
            # pass

    print('starting training', proc_id())

    # Main loop: collect experience in env and update/log each epoch
    local_steps = 0
    local_costs = 0
    local_steps_per_epoch = steps_per_epoch // num_procs()
    local_batch_size = batch_size // num_procs()
    epoch_start_time = time.time()
    flag1 = False
    for t in range(int(total_steps) // num_procs()):
        """
        Until local_start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards,
        use the learned policy.
        """
        if t > local_start_steps: # 1000
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, info = env.step(a)
        r *= reward_scale  # yee-haw
        c = info.get('cost', 0)
        ep_ret += r
        ep_cost += c
        ep_len += 1
        ep_goals += 1 if info.get('goal_met', False) else 0
        local_steps += 1
        local_costs += int(c)

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        # replay_buffer.store(o, a, r, o2, d, c)
        replay_buffer.add(o, a, r, o2, d, c, None)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len, EpGoals=ep_goals)
            o, r, d, ep_ret, ep_cost, ep_len, ep_goals = env.reset(), 0, False, 0, 0, 0, 0

        if t > 0 and t % update_freq == 0:
            for j in range(update_freq):
                batch, idxes = replay_buffer.sample_batch(local_batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             c_ph: batch['costs'],
                             d_ph: batch['done'],
                            }
                if t < local_update_after:
                    values = sess.run(vars_to_get, feed_dict)
                    logger.store(**values)
                    if j % 10 == 0 and proc_id() == 0:
                        summary = sess.run(merged_summary, feed_dict)
                        summary_writer.add_summary(summary, global_step=t*num_procs())
                else:

                    values, _ = sess.run([vars_to_get, grouped_update], feed_dict)
                    logger.store(**values)
                    if j % dual_ascent_inverval == 0 and pointwise_multiplier:
                        sess.run(train_lam_op, feed_dict)
                    if j % 10 == 0 and proc_id() == 0:
                        summary = sess.run(merged_summary, feed_dict)
                        summary_writer.add_summary(summary, global_step=(t + j) *num_procs())
            if prioritized_experience_replay:
                priority = np.abs(values['QCTDError'])
                replay_buffer.update_priorities(idxes, priority)
        # End of epoch wrap-up
        if t > 0 and t % local_steps_per_epoch == 0:
            epoch = t // local_steps_per_epoch


            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, epoch)

            # Test the performance of the deterministic version of the agent.
            test_start_time = time.time()
            test_agent()
            logger.store(TestTime=time.time() - test_start_time)

            logger.store(EpochTime=time.time() - epoch_start_time)
            epoch_start_time = time.time()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpCost', with_min_and_max=True)
            logger.log_tabular('TestEpCost', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('EpGoals', average_only=True)
            logger.log_tabular('TestEpGoals', average_only=True)
            logger.log_tabular('TotalEnvInteracts', mpi_sum(local_steps))
            logger.log_tabular('TotalCosts', mpi_sum(local_costs))
            logger.log_tabular('CostRate', mpi_sum(local_costs)/mpi_sum(local_steps))
            logger.log_tabular('QR1Vals', with_min_and_max=True)
            logger.log_tabular('QR2Vals', with_min_and_max=True)
            logger.log_tabular('QRTarget', with_min_and_max=True)
            logger.log_tabular('QCVals', with_min_and_max=True)
            if double_qc:
                logger.log_tabular('QC2Vals', with_min_and_max=True)
            logger.log_tabular('QCTarget', with_min_and_max=True)
            logger.log_tabular('QCTDError', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQR1', average_only=True)
            logger.log_tabular('LossQR2', average_only=True)
            logger.log_tabular('LossQC', average_only=True)
            if double_qc:
                logger.log_tabular('LossQC1', average_only=True)
                logger.log_tabular('LossQC2', average_only=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('LogAlpha', average_only=True)
            logger.log_tabular('Alpha', average_only=True)
            logger.log_tabular('Violation', with_min_and_max=True)
            logger.log_tabular('ViolationRate', average_only=True)
            if use_costs:
                if not pointwise_multiplier:
                    logger.log_tabular('LossBeta', average_only=True)
                    logger.log_tabular('LogBeta', average_only=True)
                    logger.log_tabular('Beta', average_only=True)
                else:
                    logger.log_tabular('Lam', with_min_and_max=True)
                    logger.log_tabular('PenaltyTerms', average_only=True)
                    logger.log_tabular('LossLam', average_only=True)
            logger.log_tabular('PiEntropy', average_only=True)
            logger.log_tabular('TestTime', average_only=True)
            logger.log_tabular('EpochTime', average_only=True)
            logger.log_tabular('TotalTime', time.time()-start_time)
            logger.dump_tabular()
    summary_writer.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--motivation', type=str, default='change env test')
    # Envs
    parser.add_argument('--env', type=str, default='Safexp-PointButton1-v0')
    parser.add_argument('--render', default=False, action='store_true')

    # Runs
    parser.add_argument('--exp_name', type=str, default='fsac')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--cpu', type=int, default=16)
    parser.add_argument('--steps_per_epoch', type=int, default=16000) # 16000
    parser.add_argument('--seed', '-s', type=int, default=0)

    # Models
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)

    # Agents
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--cost_gamma', type=float, default=0.995)
    parser.add_argument('--update_freq', type=int, default=100)
    parser.add_argument('--local_start_steps', default=1000, type=int)
    parser.add_argument('--local_update_after', default=1000, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)

    # Constrained RL
    parser.add_argument('--constrained_costs', default=True)
    parser.add_argument('--prioritized_experience_replay', default=False)
    # entropy
    parser.add_argument('--fixed_entropy_bonus', default=None, type=float)
    parser.add_argument('--entropy_constraint', type=float, default=-1.0)
    # constraint threshold
    parser.add_argument('--fixed_cost_penalty', default=None, type=float)
    parser.add_argument('--cost_constraint', type=float, default=3.)
    parser.add_argument('--cost_lim', type=float, default=None)
    # Lam net
    parser.add_argument('--pointwise_multiplier', default=True)
    parser.add_argument('--lam_lr', type=float, default=5e-6)
    parser.add_argument('--dual_ascent_interval', type=int, default=4)
    parser.add_argument('--max_lam_grad_norm', type=float, default=1.0)



    args = parser.parse_args()

    try:
        import safety_gym
    except:
        print('Make sure to install Safety Gym to use constrained RL environments.')

    mpi_fork(args.cpu, bind_to_core=True)

    from safe_rl.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    fsac(lambda: gym.make(args.env), actor_fn=mlp_actor, critic_fn=mlp_critic,
         ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
         gamma=args.gamma, cost_gamma=args.cost_gamma, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
         logger_kwargs=logger_kwargs, steps_per_epoch=args.steps_per_epoch,
         update_freq=args.update_freq, lr=args.lr, lam_lr=args.lam_lr, render=args.render,
         local_start_steps=args.local_start_steps, local_update_after=args.local_update_after,
         fixed_entropy_bonus=args.fixed_entropy_bonus, entropy_constraint=args.entropy_constraint,
         fixed_cost_penalty=args.fixed_cost_penalty, cost_constraint=args.cost_constraint,
         pointwise_multiplier=args.pointwise_multiplier, cost_lim=args.cost_lim, dual_ascent_inverval=args.dual_ascent_interval,
         max_lam_grad_norm=args.max_lam_grad_norm, motivation=args.motivation, env_id=args.env,
         constrained_costs=args.constrained_costs,
         )
