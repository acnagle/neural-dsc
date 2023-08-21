import argparse
import fire
import glob
import os
from typing import Dict

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
import seaborn as sns; sns.set(context='paper', style='white', font_scale=1.6, font='Times New Roman')
import seaborn as sns

import numpy as np
import torch

_MARKERS = ['o', '^', 's', 'x', 'm']


def _plot_points(ax, points, **kwargs):
    xs, ys = zip(*points)
    ax.plot(xs, ys, **kwargs)


def plot_iid_flip(out_file: str):
    nonlearning_results = np.loadtxt('./baseline/nonlearning/iid_flip.csv', delimiter=',')

    ldpc_points = [(nonlearning_results[0][i], nonlearning_results[1][i]) for i in range(len(nonlearning_results[0]))]
    si_only_points = [(nonlearning_results[0][i], nonlearning_results[2][i]) for i in range(len(nonlearning_results[0]))]
    msg_only_points = [(nonlearning_results[0][i], nonlearning_results[3][i]) for i in range(len(nonlearning_results[0]))]

    dirs = [os.path.join('./baseline', d) for d in os.listdir('./baseline') if os.path.isdir(os.path.join('./baseline', d)) and 'nonlearning' not in d and 'bkup' not in d]
    joint_points = []
    dist_points = []
    separate_points = []
    for d in dirs:
        ckpts = glob.glob(os.path.join(d, 'ckpt_*.pt'))
        latest_ckpt = ckpts[-1]
        dd = torch.load(latest_ckpt)
        point = (dd['args']['p_n'], dd['stats']['val_bler'][-1])
        if 'joint' in latest_ckpt:
            joint_points.append(point)
        elif 'dist' in latest_ckpt:
            dist_points.append(point)
        elif 'separate' in latest_ckpt and dd['args']['p_n'] in [0.0, 0.0048, 0.0102, 0.015, 0.0204, 0.0252, 0.03]:
            separate_points.append(point)

    plt.clf()
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1,1,1)

    _plot_points(ax, ldpc_points, label='LDPC', markersize=8, color='tab:red', zorder=50)
    _plot_points(ax, joint_points, label='Joint', markersize=8, marker='^', color='tab:orange', zorder=40)
    _plot_points(ax, dist_points, label='Distributed', markersize=8, marker='o', color='tab:blue', zorder=30)

    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_xlabel('Probability of a Bit Flip')
    ax.set_ylabel('Block Error Rate')
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 0.014)
    ax.legend(loc='lower right')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved to {out_file}')


if __name__ == '__main__':
    plot_iid_flip('paper/iid_flip.pdf')
