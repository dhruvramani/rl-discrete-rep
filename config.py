import os
import argparse

def str2bool(v):
    return v.lower() == 'true'

def str2list(v):
    if not v:
        return v
    else:
        return [v_ for v_ in v.split(',')]

def argparser():
    parser = argparse.ArgumentParser("Discrete Representation for Continuous Action Space",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # ---- SAC ----
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='half_cheetah_sac')

    args = parser.parse_args()
    return args
