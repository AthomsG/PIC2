import numpy as np
import os
from collections import deque
import tensorflow as tf

class Rqn(object):
    def __init__(self, n_ac, discount=0.99, lr=1e-4, reg='shannon', lambd=1):
        tf.reset_default_graph()
        self.vgs = tf.Variable(0, name='v_global_step', trainable=False)
        self.pgs = tf.Variable(0, name='p_global_step', trainable=False)
        self.n_ac = n_ac
        self.discount = discount
        self.reg = reg
        self.lambd = lambd
        # placeholders
        self.input = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name='inputs')
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32, name='actions')
        self.next_input = tf.placeholder(shape=[None, ], dtype=tf.float32, name='next_inputs')
        # network
        with tf.variable_scope('qnet'):
            self.qvals = self.net(self.input)
        with tf.variable_scope('target'):
            self.target_qvals = self.net(self.input)
        with tf.variable_scope('policy'):
            self.policy = self.policy_net(self.input)
        self.v_trainable_variables = tf.trainable_variables('qnet')
        self.p_trainable_variables = tf.trainable_variables('policy')
        batch_size = tf.shape(self.input)[0]
        gather_indices = tf.range(batch_size) * self.n_ac + self.actions
        self.action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)
        self.action_pi = tf.gather(tf.reshape(self.policy, [-1]), gather_indices)
        self.v_loss = tf.reduce_mean(tf.squared_difference(self.next_input, self.action_q))
        if self.reg == 'shannon':
            phi = - tf.log(self.policy+1e-8)
        elif self.reg == 'tsallis':
            phi = 0.5 * (1 - self.policy)
        elif self.reg == 'cosx':
            phi = tf.cos(0.5 * np.pi * self.policy)
        elif self.reg == 'expx':
            phi = np.exp(1) - tf.exp(self.policy)
        self.p_loss = tf.reduce_mean(tf.reduce_sum((- self.lambd * phi - self.qvals)*self.policy, axis=1))

        self.v_optimizer = tf.train.AdamOptimizer(lr)
        self.v_train_op = self.v_optimizer.minimize(self.v_loss, global_step=self.vgs,
                                                var_list=self.v_trainable_variables)
        self.p_optimizer = tf.train.AdamOptimizer(lr)
        self.p_train_op = self.p_optimizer.minimize(self.p_loss, global_step=self.pgs,
                                                var_list=self.p_trainable_variables)
        self.train_op = [self.v_train_op, self.p_train_op]                                                
        self.update_target_op = self._update_target_op()

        self.saver = tf.train.Saver(max_to_keep=50)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.update_target()

    def net(self, x):
        conv1 = tf.contrib.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(fc1, self.n_ac, activation_fn=None)
        return fc2

    def policy_net(self, x):
        conv1 = tf.contrib.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(fc1, self.n_ac, activation_fn=tf.nn.softmax)
        if self.reg == 'shannon':
            return fc2
        else:
            fc3 = tf.contrib.layers.fully_connected(fc1, self.n_ac, activation_fn=tf.nn.relu)
            fc4 = fc2 * fc3 + 1e-10
            fc5 = fc4/tf.reduce_sum(fc4, axis=1, keepdims=True)
            return fc5

    def _update_target_op(self):
        params1 = tf.trainable_variables('qnet')
        params1 = sorted(params1, key=lambda v: v.name)
        params2 = tf.trainable_variables('target')
        params2 = sorted(params2, key=lambda v: v.name)

        update_ops = []
        for param1, param2 in zip(params1, params2):
            update_ops.append(param2.assign(param1))
        return update_ops

    def update_target(self):
        self.sess.run(self.update_target_op)

    def get_action(self, obs, epsilon):
        pi = self.sess.run(self.policy, feed_dict={self.input: obs})
        batch_size = obs.shape[0]
        actions = np.array([0 for k in range(batch_size)])
        for i in range(batch_size):
            action = np.random.choice(self.n_ac, p=pi[i])
            actions[i] = action
        return actions

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        batch_size = state_batch.shape[0]
        target_next_q_vals, next_pi = self.sess.run([self.target_qvals, self.policy], feed_dict={self.input: next_state_batch})
        if self.reg == 'none':
            next_actions = np.argmax(target_next_q_vals, axis=1)
            next_phis = np.array([0 for k in range(batch_size)])
        else:
            next_actions = np.array([0 for k in range(batch_size)])
            next_phis = np.array([0 for k in range(batch_size)])
            for i in range(batch_size):
                next_action = np.random.choice(self.n_ac, p=next_pi[i])
                next_actions[i] = next_action
                if self.reg == 'shannon':
                    next_phis[i] = - np.log(next_pi[i, next_action]+1e-8)
                elif self.reg == 'tsallis':
                    next_phis[i] = 0.5 * (1 - next_pi[i, next_action])
                elif self.reg == 'cosx':
                    next_phis[i] = np.cos(0.5 * np.pi * next_pi[i, next_action])
                elif self.reg == 'expx':
                    next_phis[i] = np.exp(1) - np.exp(next_pi[i, next_action])
        targets = reward_batch + (1 - done_batch) * self.discount * (target_next_q_vals[
            np.arange(batch_size), next_actions] + self.lambd * next_phis)
        _, total_t, p_loss, v_loss = self.sess.run([self.train_op, self.vgs, self.p_loss, self.v_loss],
                                         feed_dict={self.input: state_batch, self.actions: action_batch,
                                                    self.next_input: targets})
        return total_t, {'v_loss': v_loss, 'p_loss': p_loss}

    def save_model(self, outdir):
        total_t = self.sess.run(self.vgs)
        self.saver.save(self.sess, outdir + '/model', total_t, write_meta_graph=False)

    def load_model(self, outdir):
        latest_checkpoint = tf.train.latest_checkpoint(outdir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            print("New start")

    def get_global_step(self):
        return self.sess.run(self.vgs)
