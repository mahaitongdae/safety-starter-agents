#!/usr/bin/env python

import time
import numpy as np
from safe_rl.utils.load_utils import load_policy, load_policy_withQ
from safe_rl.utils.logx import EpochLogger
from safe_rl.pg.utils import discount_cumsum
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def run_policy_withQ(env, get_action, get_values, max_ep_len=None, num_episodes=1, render=True,
                     **kwargs):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("
    dir_name = 'tests_itr_' + str(kwargs.get('itr'))
    test_dir = os.path.join(kwargs.get('log_dir'),dir_name)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    logger = EpochLogger(output_dir=test_dir)
    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    ep_qr = []
    ep_qc = []
    ep_lam = []
    ep_cost_l = []
    ep_rew_l = []
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        values = get_values(o, a)
        ep_qr.append(values['qr'])
        ep_qc.append(values['qc'])
        ep_lam.append(values['lam'])
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_cost += info.get('cost', 0)
        # c = 0.1 * ( -  info.get('cost', 0) + 1)
        ep_cost_l.append(info.get('cost', 0))
        ep_rew_l.append(r)
        ep_len += 1

        if d or (ep_len == max_ep_len):
            a = get_action(o)
            qc_terminal = get_values(o, a)['qc']
            qr_terminal = get_values(o, a)['qr']
            lam_terminal = get_values(o, a)['lam']
            ep_cost_l.append(float(qc_terminal))
            ep_rew_l.append(float(qr_terminal))
            ep_qc.append(qc_terminal)
            ep_qr.append(qr_terminal)
            ep_lam.append(lam_terminal)
            ep_true_qc = discount_cumsum(ep_cost_l, 0.99)
            ep_true_qr = discount_cumsum(ep_rew_l, 0.99)
            bias = np.array(ep_qc).squeeze()-ep_true_qc
            df = pd.DataFrame(dict(QC=np.array(ep_qc).squeeze(),
                                   TrueQC=ep_true_qc,
                                   Lam=np.array(ep_lam).squeeze(),
                                   QR=np.array(ep_qr).squeeze(),
                                   TrueQR=ep_true_qr,
                                   Steps=np.arange(ep_len + 1),
                                   Bias=bias))
            csv_name = 'runs_' + str(n) + '.csv'
            csv_path = os.path.join(test_dir, csv_name)
            df.to_csv(csv_path)
            plt.figure()
            plt.plot(df['Steps'], df['QC'], label='QC')
            plt.plot(df['Steps'], df['TrueQC'], label='realQC')
            plt.legend(loc='best')
            fig_name = 'runs_' + str(n) + '_qc.png'
            plt.savefig(os.path.join(test_dir, fig_name))
            plt.figure()
            plt.plot(df['Steps'], df['QR'], label='QR')
            plt.plot(df['Steps'], df['TrueQR'], label='realQR')
            plt.legend(loc='best')
            fig_name = 'runs_' + str(n) + '_qr.png'
            plt.savefig(os.path.join(test_dir, fig_name))
            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(n, ep_ret, ep_cost, ep_len))
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            ep_qr = []
            ep_qc = []
            ep_lam = []
            ep_cost_l = []
            ep_rew_l = []
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpCost', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

def plot(file):
    df = pd.read_csv(file)
    plt.figure()
    plt.plot(df['Steps'],df['QC'],label='QC')
    plt.plot(df['Steps'], df['TrueQC'],label='realQC')
    plt.legend(loc='best')
    plt.figure()
    plt.plot(df['Steps'], df['QR'], label='QR')
    plt.plot(df['Steps'], df['TrueQR'], label='realQR')
    plt.legend(loc='best')
    # sns.pointplot(x='Steps',y='QC', data=df)
    # sns.pointplot(x='Steps',y='TrueQC', data=df)
    plt.show()

def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default=
    '/home/mahaitong/PycharmProjects/safety-starter-agents/data/2021-03-31_fsac-per-dq_PointButton1/2021-03-31_18-20-33-fsac-per-dq_PointButton1_s0')
    parser.add_argument('--len', '-l', type=int, default=None)
    parser.add_argument('--episodes', '-n', type=int, default=5)
    parser.add_argument('--norender', '-nr', action='store_true', default=False)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true', default=False)
    args = parser.parse_args()
    model_path = os.path.join(args.fpath, 'models')
    env, get_actions, get_values, sess = load_policy_withQ(model_path,
                                                           args.itr if args.itr >= 0 else 'last',
                                                           args.deterministic)
    run_policy_withQ(env, get_actions, get_values, args.len, args.episodes, not (args.norender),
                     log_dir=args.fpath, itr=args.itr)

if __name__ == '__main__':
    run()
    # plot(
    #     '/home/mahaitong/PycharmProjects/safety-starter-agents/data/2021-03-31_fsac_per_v2_PointButton1/2021-03-31_08-38-08-fsac_per_v2_PointButton1_s0/tests/runs_0.csv')
#