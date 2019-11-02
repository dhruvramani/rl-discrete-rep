import os
import gym
import numpy as np

import sac.core as core
from sac.sac import sac
from config import argparser
from sac.utils.run_utils import setup_logger_kwargs

import discretize

def main(config):
	#logger_kwargs = setup_logger_kwargs(config.exp_name, config.seed)
	#sac(lambda : gym.make(config.env), actor_critic=core.mlp_actor_critic,
 	#ac_kwargs=dict(hidden_sizes=[config.hid] * config.l),
 	#gamma=config.gamma, seed=config.seed, epochs=config.epochs,
 	#logger_kwargs=logger_kwargs)

 	discretize.discretize_policy(config, gym.make(config.env))


if __name__ == '__main__':
	config = argparser()
	main(config)	
