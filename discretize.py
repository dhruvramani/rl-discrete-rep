import os
import gym
import numpy as np
import tensorflow as tf

import sac.core as core
from sac.sac import sac
from gumbel_softmax import *

slim = tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli

# def dummy_discrete(config, env):
# 	ac_kwargs['action_space'] = env.action_space
# 	x_ph, a_ph = core.placeholders(obs_dim, act_dim)
# 	    with tf.variable_scope('main'):
#         mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

def get_data(config, step):
	for _ in range(config.max_iters):
		data = []
		for _ in range(config.batch_size):
			data.append(step())
		yield np.asarray(data)

def policy_step(mu, pi, x_ph):
	def get_action(o, deterministic=False):
		act_op = mu if deterministic else pi
		return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]
	return get_action 

def discretize_policy(config, env, policy=False):
	obs = env.reset()
	K = config.n_classes
	N = config.n_cat_dist
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	x_ph, a_ph = core.placeholders(obs_dim, act_dim)
	if(policy == True):
		ac_kwargs['action_space'] = env.action_space
		mu, pi = core.actor_critic(x_ph, a_ph, **ac_kwargs)
		step = policy_step(mu, pi, x_ph)
	else :
		step = env.action_space.sample

	# ---- Encoder ----
	net = slim.stack(a_ph, slim.fully_connected,[act_dim, 64])
	# unnormalized logits for N separate K-categorical distributions (shape=(batch_size*N,K))
	logits_y = tf.reshape(slim.fully_connected(net, K*N, activation_fn=None),[-1, K])
	q_y = tf.nn.softmax(logits_y)
	log_q_y = tf.log(q_y + 1e-20)
	
	# ---- Decoder ----
	tau = tf.Variable(5.0, name="temperature")
	# sample and reshape back (shape=(batch_size,N,K))
	# set hard=True for ST Gumbel-Softmax
	y = tf.reshape(gumbel_softmax(logits_y, tau, hard=False),[-1, N, K])
	# generative model p(x|y), i.e. the decoder (shape=(batch_size,200))
	net = slim.stack(slim.flatten(y), slim.fully_connected, [512, 256])
	logits_x = slim.fully_connected(net, act_dim, activation_fn=None)
	# (shape=(batch_size, act_dim))
	p_x = Bernoulli(logits=logits_x)

	# loss and train ops
	kl_tmp = tf.reshape(q_y * (log_q_y - tf.log(1.0 / K)), [-1, N, K])
	KL = tf.reduce_sum(kl_tmp, [1, 2])
	elbo = tf.reduce_sum(p_x.log_prob(a_ph), 1) - KL

	loss = tf.reduce_mean(-elbo)
	lr = tf.constant(config.lr)
	train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=slim.get_model_variables())
	init_op = tf.initialize_all_variables()

	sess = tf.InteractiveSession()
	sess.run(init_op)

	if(not os.path.isfile(config.gumbel_path)):
		tau0=1.0 # initial temperature
		np_temp = tau0
		np_lr = config.lr

		dat = []
		saver = tf.train.Saver()
		for i, np_x in enumerate(get_data(config, step)):
			_, np_loss = sess.run([train_op, loss], {a_ph:np_x, tau:np_temp, lr:np_lr})
			if i % 100 == 1:
				dat.append([i, np_temp, np_loss])
			if i % 1000 == 1:
		  		save_path = saver.save(sess, config.gumbel_path)
		  		np_temp = np.maximum(tau0 * np.exp(- config.anneal_rate * i), config.min_temp)
		  		np_lr *= 0.9
			if i % 5000 == 1:
		  		print('Step %d, ELBO: %0.3f' % (i, -np_loss))
	else:
		saver.restore(sess, config.gumbel_path)

	for _ in range(50):
		y_pred = sess.run(y, {a_ph : step()})
		print(y_pred)


def discretize_main(config, env):
	obs = env.reset()
	print(env.action_space.sample())
