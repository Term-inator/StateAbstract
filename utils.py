import math

import numpy as np
import pandas as pd
import umap
from matplotlib import pyplot as plt
from scipy.sparse import lil_matrix
from scipy.special import gamma, factorial, comb
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS, Isomap, SpectralEmbedding


def three_sigma(series):
    """
    series：表示传入DataFrame的某一列。
    """
    series = series[~((series == 0) | (series == 1))]
    rule = (series.mean() - 3 * series.std() > series) | (series.mean() + 3 * series.std() < series)
    index = np.arange(series.shape[0])[rule]
    return index  # 返回落在3sigma之外的行索引值


def delete_out_three_sigma(data):
    """
    data：待检测的DataFrame
    """
    out_index = []  # 保存要删除的行索引
    col = [data.columns.get_loc('x'), data.columns.get_loc('y')]
    for j in col:  # 对每一列分别用3sigma原则处理
        index = three_sigma(data.iloc[:, j])
        print(data.iloc[:, j][index])
        out_index += index.tolist()
    delete_ = list(set(out_index))
    # print('所删除的行索引为：', delete_)
    data.drop(delete_, inplace=True)
    return data


def delete_threshold(data, threshold=10):
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


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# 全局变量，用于记忆化搜索
memo = None


def init_memo(K_max, n, m):
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


def calc_count(K, n, m):
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


def calc_prob(K, n, m):
    """
    计算从 K 个不同数字中可放回地取出 n 个，出现 m 种数字的概率
    """
    return 1 - calc_count(K, n, m) / K ** n


def stirling(n, m):
    res = 0
    for i in range(m+1):
        res += pow(-1, m-i) * pow(i, n) / (factorial(i, exact=True) * factorial(m-i, exact=True))
    return res


def calc_count_opt(K, n, m):
    res = stirling(n, m)
    for i in range(0, m):
        res *= (K-i)
    return round(res)


def opt_correct_prove():
    init_memo(40, 40, 40)
    for K in range(2, 40):
        for n in range(2, min(K+1, 15)):
            for m in range(2, n+1):
                count = calc_count(K, n, m)
                count_opt = calc_count_opt(K, n, m)
                if count != 0 and abs(count - count_opt)/count > 1e-6:
                    print(f'K = {K}, n = {n}, m = {m}, wrong, ori:{calc_count(K, n, m)}, opt:{calc_count_opt(K, n, m)}')
                else:
                    print(f'K = {K}, n = {n}, m = {m}, correct, eps={abs(count - count_opt)}')
            print()
        print()
    print()


def delta():
    init_memo(60, 60, 60)
    for K in range(2, 60):
        for n in range(2, min(K+1, 15)):
            for m in range(2, n+1):
                if calc_count(K, n, m) > calc_count(K, n, m-1):
                    print('K = {}, n = {}, m = {}, inc'.format(K, n, m))
                elif calc_count_opt(K, n, m) < calc_count_opt(K, n, m-1):
                    print('K = {}, n = {}, m = {}, dec'.format(K, n, m))
                else:
                    print('K = {}, n = {}, m = {}, same'.format(K, n, m))
            print()
        print()
    print()


def cluster_visualize(model, data, display_type='tsne', n_components=2, display_size='normal'):
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
        raise ValueError('display_type must be one of tsne, pca, lle, mds, isomap, umap, spectral, lda')

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
        plt.scatter(data_vis[:, 0], data_vis[:, 1], c=model.labels_, s=s, alpha=alpha, edgecolors='none')
        plt.show()
    elif n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_vis[:, 0], data_vis[:, 1], data_vis[:, 2], c=model.labels_, s=s, alpha=alpha, edgecolors='none')
        plt.show()


def nearest_multiple(original_value, divisor, round_type='round'):
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


def expand_action_range(action_range, granularity):
    return nearest_multiple(action_range[0], granularity, round_type='floor'), \
        nearest_multiple(action_range[1], granularity, round_type='ceil')
