"""
display npy to plot
"""

import os
import numpy as np
import matplotlib.pyplot as plt


task = 'intersection'
directory = f'output/{task}/env_policy'


def get_data(npy_file):
    data = np.load(npy_file)
    return data


def try_K_plot(sse, silhouette, calinski_harabasz, steady, save=False):
    fig1 = plt.figure(figsize=(15, 8))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    # ax1.set_ylabel('SSE', fontsize=20)
    ax1.set_xlabel('K', fontsize=20)
    ax1.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)

    fig2 = plt.figure(figsize=(15, 8))
    ax3 = fig2.add_subplot(1, 1, 1)
    ax3.set_xlabel('K', fontsize=20)

    ax1.plot(sse[0], sse[1], 'b', label='SSE')
    ax2.plot(silhouette[0], silhouette[1], 'g', label='Silhouette')
    ax2.plot(calinski_harabasz[0], calinski_harabasz[1], 'y', label='Calinski-Harabasz')
    ax3.plot(steady[0], steady[1], 'b', label='Steady')
    ax2.plot(steady[0], steady[1], 'r', label='Steady')

    plt.xticks(range(sse[0][0], sse[0][-1] + 1, 2))

    ax1.legend(loc='upper left', fontsize=20)
    ax2.legend(loc='upper right', fontsize=20)
    ax3.legend(loc='upper left', fontsize=20)

    ax3.tick_params(labelsize=20)

    plt.show()

    if save:
        # fig1.savefig(os.path.join(directory, 'plot_sse_sh_ch.png'))
        fig2.savefig(os.path.join(directory, 'plot_steady.png'))
        fig1.savefig(os.path.join(directory, 'plot_all.png'))


def episode_reward_plot(reward, episode_reward, save=False):
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)

    ax.plot(reward[0], reward[1], 'b', label='reward')
    K = list(reward[0])
    K.sort()
    plt.xticks(range(K[0], K[-1]+1, 2))
    # for x, y in zip(reward[0], reward[1]):
    #     ax.text(x, y, f'{y:.2f}')
    plt.axhline(y=episode_reward, linestyle='--')
    # ax.text(0, episode_reward, f'{episode_reward:.2f}')
    ax.set_xlabel('K')
    ax.set_ylabel('Reward')
    # plt.title('Episode Reward')

    plt.show()

    if save:
        fig.savefig(os.path.join(directory, 'plot_episode_reward.png'))


def get_sse(file):
    data = get_data(file)
    K, sse = zip(*data)
    K = list(K)
    for i in range(len(K)):
        K[i] = int(K[i])
    return K, sse


def get_silhouette(file):
    data = get_data(file)
    K, silhouette = zip(*data)
    K = list(K)
    for i in range(len(K)):
        K[i] = int(K[i])
    return K, silhouette


def get_calinski_harabasz(file):
    data = get_data(file)
    K, calinski_harabasz = zip(*data)
    K = list(K)
    for i in range(len(K)):
        K[i] = int(K[i])
    return K, calinski_harabasz


def get_prism_experiment(file):
    data = get_data(file)
    K, acc, steer, reward = zip(*data)
    K = list(K)
    for i in range(len(K)):
        K[i] = int(K[i])
    return K, acc, steer, reward


def comparison(rewards, episode_reward, K_step, title, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(rewards)):
        ax.plot(rewards[i][0], rewards[i][1], label=f'{rewards[i][2]}')
    K = list(rewards[0][0]) if len(rewards[0][0]) > len(rewards[1][0]) else list(rewards[1][0])
    K.sort()
    plt.xticks(range(K[0], K[-1]+1, K_step), fontsize=14)
    plt.yticks(fontsize=14)
    plt.axhline(y=episode_reward, linestyle='--')
    ax.set_xlabel('K', fontsize=14)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=14)
    # plt.subplots_adjust(left=10, right=2, top=2, bottom=10)

    plt.show()

    if save:
        fig.savefig(os.path.join(f'plot_episode_reward_comparison - {title}.png'))


if __name__ == '__main__':
    # file_sse = os.path.join(directory, 'sse.npy')
    # x_sse, value_sse = get_sse(file_sse)
    #
    # file_sil = os.path.join(directory, 'silhouette.npy')
    # x_sil, value_sil = get_silhouette(file_sil)
    #
    # file_ch = os.path.join(directory, 'calinski_harabasz.npy')
    # x_ch, value_ch = get_calinski_harabasz(file_ch)
    # print(np.vstack((x_sse, value_sse, value_sil, value_ch)).T)
    #
    # file_st = os.path.join(directory, 'steady.npy')
    # x_st, value_st = get_sse(file_st)
    # print(np.vstack((x_st, value_st)).T)
    #
    # try_K_plot((x_sse, value_sse), (x_sil, value_sil), (x_ch, value_ch), (x_st, value_st), save=True)

    # file = os.path.join(directory, 'record.npy')
    # K, _, _, reward = get_prism_experiment(file)
    # print(np.vstack((K, reward)).T)
    # episode_reward_plot((K, reward), 11.02, save=True)

    experiments = {
        'acc': [
            {
                'dirs': ['cmp_policy', 'env_policy'],
                'labels': ['no curiosity', 'use curiosity'],
                'K_step': 2,
                'episode_reward': 59.94
            },
            {
                'dirs': ['cmp_policy3', 'env_policy3'],
                'labels': ['no curiosity', 'use curiosity'],
                'K_step': 2,
                'episode_reward': 241.90
            }
        ],
        'lane_keeping': [
            {
                'dirs': ['cmp_policy', 'env_policy'],
                'labels': ['no curiosity', 'use curiosity'],
                'K_step': 3,
                'episode_reward': 48.50
            },
            {
                'dirs': ['cmp_policy2', 'env_policy2'],
                'labels': ['no curiosity', 'use curiosity'],
                'K_step': 3,
                'episode_reward': 46.40
            }
        ],
        'intersection': [
            {
                'dirs': ['cmp_policy', 'env_policy'],
                'labels': ['no curiosity', 'use curiosity'],
                'K_step': 2,
                'episode_reward': 11.02
            },
            {
                'dirs': ['cmp_policy2', 'env_policy2'],
                'labels': ['no curiosity', 'use curiosity'],
                'K_step': 2,
                'episode_reward': 9.36
            }
        ]
    }

    for task in experiments:
        for i, exp in enumerate(experiments[task]):
            dirs = exp['dirs']
            labels = exp['labels']
            episode_reward = exp['episode_reward']
            rewards = []
            for j, d in enumerate(dirs):
                file = os.path.join('output', task, d, 'record.npy')
                K, _, _, reward = get_prism_experiment(file)
                rewards.append((K, reward, labels[j]))
            title = f'{task} - policy {i + 1}'
            comparison(rewards, episode_reward, K_step=exp['K_step'], title=title, save=True)
