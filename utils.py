import numpy as np
import pandas as pd

from State import Trajectory


def read_csv(filename):
    data_csv = pd.read_csv(filename)

    # data_csv = delete_out_three_sigma(data_csv)
    data_csv = delete_threshold(data_csv, threshold=20)

    data = []
    for i in data_csv.index:
        trajectory = Trajectory()
        # print(data_csv.loc[i].values[0:-1])
        trajectory.load(data_csv.loc[i])
        # print(state.__repr__())
        data.append(trajectory)

    return np.array(data)


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