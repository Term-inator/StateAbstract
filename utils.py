import copy
import importlib
import math
from typing import Callable

import numpy as np
import pandas as pd
import umap
import yaml
from matplotlib import pyplot as plt
from scipy.sparse import lil_matrix
from scipy.special import gamma, factorial, comb
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS, Isomap, SpectralEmbedding
from tqdm import tqdm


def three_sigma(series):
    """
    TODO 计划删除
    series：表示传入DataFrame的某一列。
    """
    series = series[~((series == 0) | (series == 1))]
    rule = (series.mean() - 3 * series.std() > series) | (series.mean() + 3 * series.std() < series)
    index = np.arange(series.shape[0])[rule]
    return index  # 返回落在3sigma之外的行索引值


def delete_out_three_sigma(data):
    """
    TODO 计划删除
    data：待检测的DataFrame
    """
    out_index = []  # 保存要删除的行索引
    col = [0, 1]
    for j in col:  # 对每一列分别用3sigma原则处理
        index = three_sigma(data.iloc[:, j])
        print(data.iloc[:, j][index])
        out_index += index.tolist()
    delete_ = list(set(out_index))
    # print('所删除的行索引为：', delete_)
    data.drop(delete_, inplace=True)
    return data


def delete_threshold(data, threshold=10):
    """
    TODO 计划删除
    """
    zero_index = np.arange(data['x'].shape[0])[data['x'] == 0]
    one_index = np.arange(data['x'].shape[0])[data['x'] == 1]
    n = len(zero_index)

    out_index = []  # 保存要删除的行索引
    for i in range(n):
        if one_index[i] - zero_index[i] <= threshold:
            for j in range(zero_index[i], one_index[i] + 1):
                out_index.append(j)

    data.drop(out_index, inplace=True)
    return data


def normalize(data, _min: float = None, _max: float = None):
    """
    归一化
    对于环境轨迹数据，参数 _min 和 _max 为 None
    对于策略轨迹数据，参数 _min 和 _max 为对应环境轨迹数据的最小值和最大值
    :param _min: 最小值
    :param _max: 最大值
    :return: 归一化后的数据、最小值和最大值
    TODO 尚不支持数组和 tqdm
    """
    norm_n = len(data[0].state.to_list())

    if _min is None and _max is None:
        _min = [100000 for _ in range(norm_n)]
        _max = [-100000 for _ in range(norm_n)]
        for j in range(norm_n):
            _sum = 0
            for traj in data:
                if traj.state.state_type != 0 and traj.state.state_type != 1:
                    if type(traj.state.to_list()[j]) == np.ndarray:
                        _min[j] = min(_min[j], min(traj.state.to_list()[j]))
                        _max[j] = max(_max[j], max(traj.state.to_list()[j]))
                    else:
                        _min[j] = min(_min[j], traj.state.to_list()[j])
                        _max[j] = max(_max[j], traj.state.to_list()[j])

    for i in range(len(data)):
        if data[i].state.state_type != 0 and data[i].state.state_type != 1:
            normed_state = []
            state = data[i].state.to_list()
            for j in range(norm_n):
                if _max[j] == _min[j]:
                    normed_state.append(state[j])
                else:
                    normed_state.append((state[j] - _min[j]) / (_max[j] - _min[j]))
            data[i].state.from_list(normed_state)

    return data, _min, _max


def standardize(data, _mean: float = None, _std: float = None) -> tuple:
    """
    标准化
    对于环境轨迹数据，参数 _mean 和 _std 为 None
    对于策略轨迹数据，参数 _mean 和 _std 为对应环境轨迹数据的均值和标准差
    :param _mean: 均值
    :param _std: 标准差
    :return: 标准化后的数据、均值和标准差
    """
    norm_n = len(data[0].state.to_list())

    if _mean is None or _std is None:
        _mean = [0 for _ in range(norm_n)]
        _std = [0 for _ in range(norm_n)]
        for j in range(norm_n):
            column = []
            for traj in data:
                if traj.state.state_type != 0 and traj.state.state_type != 1:
                    column.append(traj.state.to_list()[j])

            if type(column[0]) == np.ndarray:
                mean_matrix = np.mean(column, axis=0)

                # 计算每个矩阵的方差
                variance_matrix = np.var(column, axis=0, ddof=1)

                # 计算矩阵数组的标准差
                std_matrix = np.sqrt(variance_matrix)
                _mean[j] = mean_matrix
                _std[j] = std_matrix
            else:
                _mean[j] = np.mean(column, axis=0)
                _std[j] = np.std(column, axis=0, ddof=1)

    with tqdm(total=len(data)) as pbar:
        pbar.set_description('Standardize')
        for i in range(len(data)):
            if data[i].state.state_type != 0 and data[i].state.state_type != 1:
                normed_state = []
                state = data[i].state.to_list()
                # print(1, state, data[i].state.state_type)
                for j in range(norm_n):
                    if type(_std[j]) is np.ndarray and (_std[j] == 0).any():
                        normed_state.append(state[j])
                    elif type(_std[j]) is not np.ndarray and _std[j] == 0:
                        normed_state.append(state[j])
                    else:
                        normed_state.append((state[j] - _mean[j]) / _std[j])
                data[i].state.from_list(normed_state)
                # print(2, data[i].state.to_list())
            pbar.update(1)

    return data, _mean, _std


def load_yml(path: str) -> dict:
    """
    加载 yml 文件
    :param path: 文件路径
    :return:
    """
    with open(path, 'r', encoding='utf-8') as rf:
        config = yaml.load(rf.read(), Loader=yaml.FullLoader)
    return config


# 全局变量，用于记忆化搜索
memo = None


def init_memo(K_max: int, n: int, m: int):
    """
    初始化 memo 数组
    """
    global memo
    # memo = np.full((K_max + 1, n + 1, m + 1), -1)
    memo = []
    for i in range(K_max + 1):
        memo.append([])
        for j in range(n + 1):
            memo[i].append([])
            for k in range(m + 1):
                memo[i][j].append(-1)


def calc_count(K: int, n: int, m: int) -> int:
    """
    计算从 K 个不同数字中可放回地取出 n 个，出现 m 种数字 有多少种
    """
    if n == 0:
        return 1 if m == 0 else 0
    if m > n:
        return 0

    global memo
    if memo[K][n - 1][m] == -1:
        memo[K][n - 1][m] = calc_count(K, n - 1, m)
    c1 = memo[K][n - 1][m] * m

    if memo[K][n - 1][m - 1] == -1:
        memo[K][n - 1][m - 1] = calc_count(K, n - 1, m - 1)
    c2 = memo[K][n - 1][m - 1] * (K - m + 1)

    return c1 + c2


def calc_prob(K: int, n: int, m: int) -> float:
    """
    计算从 K 个不同数字中可放回地取出 n 个，出现 m 种数字的概率，并用 1 减
    """
    return 1 - calc_count(K, n, m) / K ** n


def stirling(n: int, m: int) -> int:
    """
    计算第二类斯特林数
    把 n个不同的数划分为 m个集合的方案数，要求不能为空集，不考虑顺序
    :param n:
    :param m:
    :return:
    """
    res = 0
    for i in range(m + 1):
        res += pow(-1, m - i) * pow(i, n) / (factorial(i, exact=True) * factorial(m - i, exact=True))
    return res


def calc_count_opt(K: int, n: int, m: int) -> int:
    """
    calc_count 函数的优化版
    直接通过通项计算
    """
    res = stirling(n, m)
    for i in range(0, m):
        res *= (K - i)
    return round(res)


def opt_correct_prove() -> None:
    """
    证明 calc_count_opt 和 calc_count 等价
    :return:
    """
    init_memo(40, 40, 40)
    for K in range(2, 40):
        for n in range(2, min(K + 1, 15)):
            for m in range(2, n + 1):
                count = calc_count(K, n, m)
                count_opt = calc_count_opt(K, n, m)
                if count != 0 and abs(count - count_opt) / count > 1e-6:
                    print(f'K = {K}, n = {n}, m = {m}, wrong, ori:{calc_count(K, n, m)}, opt:{calc_count_opt(K, n, m)}')
                else:
                    print(f'K = {K}, n = {n}, m = {m}, correct, eps={abs(count - count_opt)}')
            print()
        print()
    print()


def delta() -> None:
    """
    查看 calc_count 结果的单调性
    :return:
    """
    init_memo(60, 60, 60)
    for K in range(2, 60):
        for n in range(2, min(K + 1, 15)):
            for m in range(2, n + 1):
                if calc_count(K, n, m) > calc_count(K, n, m - 1):
                    print('K = {}, n = {}, m = {}, inc'.format(K, n, m))
                elif calc_count(K, n, m) < calc_count(K, n, m - 1):
                    print('K = {}, n = {}, m = {}, dec'.format(K, n, m))
                else:
                    print('K = {}, n = {}, m = {}, same'.format(K, n, m))
            print()
        print()
    print()


def cluster_visualize(model, data, display_type: str='tsne', n_components: int=2, display_size: str='normal') -> None:
    """
    聚类可视化
    :param model: 聚类后的模型
    :param data: 数据
    :param display_type: 降维方法
    :param n_components: 降到几维
    :param display_size: 图像中点的大小
    :return:
    """
    if display_type == 'tsne':
        # 进行t-SNE转换
        # 用于高维数据的降维和可视化。特别适用于聚类和类别之间的可视化差异。
        # 计算复杂度较高，难以可视化全局结构。
        tsne = TSNE(n_components=n_components, random_state=0, n_jobs=20)
        data_vis = tsne.fit_transform(data)
    elif display_type == 'pca':
        # 进行PCA转换
        # 用于线性降维和可视化，通常用于去除高维数据的冗余信息。可以在数据的主要方向上进行压缩，并在保留尽可能多的信息的同时减少噪音。
        # 可能会丢失一些非线性信息，如数据中存在非线性关系，PCA可能无法很好地捕捉这些关系。
        pca = PCA(n_components=n_components)
        data_vis = pca.fit_transform(data)
    elif display_type == 'lle':
        # 进行LLE转换
        # 用于保持局部距离关系的非线性降维。在局部上保持了数据的线性结构，并且可以有效地减少高维数据的维度。
        # 对于数据的全局结构不太敏感，容易出现陷入局部最小值的问题。
        lle = LocallyLinearEmbedding(n_components=n_components)
        data_vis = lle.fit_transform(data)
    elif display_type == 'mds':
        # 进行MDS转换
        # 用于保留样本间距离关系的降维算法。通常用于探索高维数据中的全局结构和相似性关系。
        # 计算复杂度高，对于大规模数据集来说非常耗时。
        mds = MDS(n_components=n_components)
        data_vis = mds.fit_transform(data)
    elif display_type == 'isomap':
        # 进行ISOMAP转换
        # 用于非线性降维，通常在高维数据中保留流形结构。ISOMAP对于数据的局部非线性结构保持得比较好。
        # 计算复杂度高，需要处理高维度矩阵，耗费大量的计算资源。
        _data = lil_matrix(data)
        isomap = Isomap(n_components=n_components)
        data_vis = isomap.fit_transform(_data)
    elif display_type == 'umap':
        # 进行UMAP转换
        # 用于高维数据降维和可视化。UMAP旨在保留数据的全局结构，同时还可以处理复杂的非线性结构。
        # 超参数调整较为复杂，需要花费较长的时间来确定最佳参数。
        umap_emb = umap.UMAP(n_components=n_components)
        data_vis = umap_emb.fit_transform(data)
    elif display_type == 'spectral':
        # 进行Spectral Embedding转换
        # 用于非线性降维，将数据映射到低维空间，同时保留了数据的局部关系。与其他算法相比，Spectral Embedding具有更好的可扩展性和更高的灵活性。
        # 需要处理高维矩阵，计算复杂度高，对于大规模数据集需要更多的时间和计算资源。
        spectral = SpectralEmbedding(n_components=n_components)
        data_vis = spectral.fit_transform(data)
    elif display_type == 'lda':
        # 进行LDA转换
        # 用于分类任务的降维算法，通过最大化类别间的方差和最小化类别内的方差，实现降维并保留类别信息。
        # 只适用于分类任务，如果数据集不是分类问题，则不能使用该算法。
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        data_vis = lda.fit_transform(data)
    else:
        data_vis = data

    if display_size == 'small':
        s = 5
        alpha = 0.3
    elif display_size == 'normal':
        s = 15
        alpha = 0.5
    else:
        raise ValueError('display_size must be one of small, normal')

    if n_components == 2:
        # 绘制2D图形，使用不同颜色表示不同的聚类
        fig = plt.figure(figsize=(15, 8))
        plt.scatter(data_vis[:, 0], data_vis[:, 1], c=model.labels_, s=s, alpha=alpha, edgecolors='none')
        plt.show()
        fig.savefig('cluster_visualize.png')
    elif n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_vis[:, 0], data_vis[:, 1], data_vis[:, 2], c=model.labels_, s=s, alpha=alpha, edgecolors='none')
        plt.show()
        fig.savefig('cluster_visualize.png')


def nearest_multiple(original_value: float, divisor: float, round_type: str='round') -> int:
    """
    根据 round_type 找到离 original_value 的最近的 divisor 的倍数
    :param original_value: 初始值
    :param divisor: 除数
    :param round_type: 近似方法
    :return:
    """
    # 确保除数为正数
    divisor = abs(divisor)

    # 找到最接近原始值的能被除数整除的数
    if round_type == 'round':
        nearest = round(original_value / divisor) * divisor
    elif round_type == 'ceil':
        nearest = math.ceil(original_value / divisor) * divisor
    elif round_type == 'floor':
        nearest = math.floor(original_value / divisor) * divisor
    else:
        raise ValueError('round_type must be one of round, ceil, floor')

    return nearest


def expand_action_range(action_range: list, granularity: float) -> tuple:
    """
    根据 granularity 扩展 action_range 为 granularity 的整数倍
    :param action_range: 动作空间的初始范围
    :param granularity: 离散化粒度
    :return:
    """
    # 下界向下扩展，上界向上扩展
    return nearest_multiple(action_range[0], granularity, round_type='floor'), \
        nearest_multiple(action_range[1], granularity, round_type='ceil')


def get_info_from_data(data: list, props_dict: dict) -> dict:
    """
    计算每个 episode 的平均步数、平均累计奖励、（超出车道概率、碰撞概率、抵达终点概率）
    :param data:
    :return:
    """
    episode = 0
    steps = 0
    num_crash = 0
    num_outoflane = 0
    num_reachdest = 0
    episode_rewards = 0

    is_crash = 0
    is_outoflane = 0
    is_reachdest = 0
    for i in range(len(data)):
        if data[i].state.state_type == 0:
            episode += 1
            is_crash = 0
            is_outoflane = 0
            is_reachdest = 0
        elif data[i].state.state_type == 1:
            num_crash += is_crash
            num_outoflane += is_outoflane
            num_reachdest += is_reachdest
        else:
            steps += 1
            episode_rewards += data[i].state.reward
            if hasattr(data[i].state, 'is_crash'):
                if data[i].state.is_crash != 0:
                    is_crash = 1
            if hasattr(data[i].state, 'is_outoflane'):
                if data[i].state.is_outoflane != 0:
                    is_outoflane = 1
            if hasattr(data[i].state, 'is_reachdest'):
                if data[i].state.is_reachdest != 0:
                    is_reachdest = 1

    avg_step = steps / episode
    avg_episode_reward = episode_rewards / episode
    p_crash = num_crash / episode
    p_outoflane = num_outoflane / episode
    p_reachdest = num_reachdest / episode

    print(f'steps: {steps}, avg_step: {avg_step}, episode: {episode}, avg_episode_reward: {avg_episode_reward}')

    res = {'avg_step': int(round(avg_step, 0))}
    if 'episode_reward' in props_dict['name']:
        res['episode_reward'] = avg_episode_reward
    if 'is_crash' in props_dict['name']:
        res['is_crash'] = p_crash
    if 'is_outoflane' in props_dict['name']:
        res['is_outoflane'] = p_outoflane
    if 'is_reachdest' in props_dict['name']:
        res['is_reachdest'] = p_reachdest

    return res


def compare_result_error(ground_truth: dict, result: dict, props_dict: dict) -> dict:
    """
    比较 ground_truth 和 result 的误差
    :param ground_truth: 性质真实值的字典
    :param result: 性质实验结果的字典
    :param props_dict: 要比较的性质
    :return:
    """
    error = {
        'absolute': {},  # 绝对误差
        'relative': {}  # 相对误差
    }
    for key in props_dict['name']:
        error['absolute'][key] = result[key] - ground_truth[key]
        if error['absolute'][key] == 0:
            error['relative'][key] = 0
        else:
            if ground_truth[key] != 0:
                error['relative'][key] = error['absolute'][key] / ground_truth[key]
            else:
                error['relative'][key] = -1
    return error


def class_from_path(path: str) -> Callable:
    """
    从路径获取类
    :param path: 类的路径
    :return: 类
    """
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object
