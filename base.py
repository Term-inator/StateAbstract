import enum
import math
import threading
from queue import Queue

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils


def read_csv(config_data, data_type, parallel=False):
    if parallel:
        def read_csv_chunk(csv_path, chunk_start, chunk_size, queue):
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size, skiprows=chunk_start):
                queue.put((chunk_start, chunk))

        def read_csv_multithread(csv_path, num_threads):
            # 获取原始CSV文件的总行数
            with open(csv_path, 'r') as f:
                num_rows = sum(1 for line in f)

            # 计算每个数据块的开始和结束位置
            chunk_size = math.ceil(num_rows / num_threads)
            chunk_starts = [i * chunk_size for i in range(num_threads)]

            queue = Queue()
            threads = []
            for i in range(num_threads):
                t = threading.Thread(target=read_csv_chunk, args=(csv_path, chunk_starts[i], chunk_size, queue))
                threads.append(t)
                t.start()

            data = {}
            for i in range(num_threads):
                chunk_start, chunk = queue.get()
                data[chunk_start] = chunk

            for t in threads:
                t.join()

            # 获取列名
            headers = data[0].columns
            data[0] = data[0][1:]
            # 按照数据块的开始位置对字典进行排序
            data = sorted(data.items())

            # 将所有数据块合并为一个DataFrame
            dataframes = []
            for _, chunk in data:
                chunk.columns = headers
                dataframes.append(chunk)
            res = pd.concat(dataframes)

            # 重置索引以避免索引冲突
            res.reset_index(drop=True, inplace=True)

            return res

        num_threads = 4
        data_csv = read_csv_multithread(config_data[data_type]['path'], num_threads)
    else:
        data_csv = pd.read_csv(config_data[data_type]['path'])

        # data_csv = utils.delete_out_three_sigma(data_csv)
        # data_csv = delete_threshold(data_csv, threshold=20)

    if 'episode_num' in config_data[data_type]:
        episode_num = config_data[data_type]['episode_num']
        parallel = False
    else:
        episode_num = len(data_csv)

    if parallel:
        def load_data(data_csv, chunk_start, chunk_size, queue, pbar):
            data = []
            for i in range(chunk_start, chunk_start + chunk_size):
                trajectory = Trajectory(config_data)
                # print(data_csv.loc[i].values[0:-1])
                trajectory.load(data_csv.loc[i])
                # print(state.__repr__())
                data.append(trajectory)
                pbar.update(1)
            queue.put((chunk_start, data))

        def load_data_multithread(data_csv, num_threads):
            # 获取原始CSV文件的总行数
            num_rows = len(data_csv)

            # 计算每个数据块的开始和结束位置
            chunk_size = math.ceil(num_rows / num_threads)
            chunk_starts = [i * chunk_size for i in range(num_threads)]
            last_chunk_size = num_rows - chunk_starts[-1]

            with tqdm(total=num_rows) as pbar:
                pbar.set_description('Loading csv data to Trajectory (parallel)')
                queue = Queue()
                threads = []
                for i in range(num_threads):
                    t = threading.Thread(target=load_data,
                                         args=(data_csv, chunk_starts[i],
                                               last_chunk_size if i == num_threads - 1 else chunk_size, queue, pbar))
                    threads.append(t)
                    t.start()

                data = {}
                for i in range(num_threads):
                    chunk_start, chunk = queue.get()
                    data[chunk_start] = chunk

                for t in threads:
                    t.join()

            # 按照数据块的开始位置对字典进行排序
            data = sorted(data.items())

            res = np.concatenate([chunk for _, chunk in data])

            return res

        num_threads = 4
        data = load_data_multithread(data_csv, num_threads)
        return data
    else:
        episode_count = 0
        data = []
        with tqdm(total=len(data_csv)) as pbar:
            pbar.set_description('Loading csv data to Trajectory')
            for i in data_csv.index:
                trajectory = Trajectory(config_data)
                # print(data_csv.loc[i].values[0:-1])
                trajectory.load(data_csv.loc[i])
                # print(trajectory.state.on_road)
                # print(state.__repr__())
                data.append(trajectory)
                pbar.update(1)

                if trajectory.state.state_type == 1:
                    episode_count += 1

                if episode_count == episode_num:
                    break

    return np.array(data)


def mean(lst):
    n = len(lst)
    if n == 0:
        return None
    elif n == 1:
        return lst[0]
    else:
        lst_mean = lst[0]
        for i in range(1, n):
            lst_mean = lst_mean + lst[i]
        lst_mean /= n
        lst_mean.tag = lst[0].tag
        return lst_mean


e_speed = 10
e_distance = 20
safe_distance = 10


class Base:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        # 不参与运算
        self.readonly = ['cluster_exclude', 'readonly']
        # 不参与聚类
        self.cluster_exclude = []
        self.cluster_exclude.extend(self.readonly)

    def __add__(self, other):
        result = Base()
        for key, value in vars(self).items():
            if key in self.readonly:
                setattr(result, key, vars(self)[key])
            else:
                if key in vars(other):
                    setattr(result, key, value + vars(other)[key])
                else:
                    setattr(result, key, value)
        for key, value in vars(other).items():
            if key not in vars(self):
                setattr(result, key, value)
        return result

    def __truediv__(self, other):
        result = Action()
        if isinstance(other, (int, float)):
            for key, value in vars(self).items():
                if key in self.readonly:
                    setattr(result, key, vars(self)[key])
                else:
                    setattr(result, key, value / other)
            return result
        else:
            return NotImplemented

    def __hash__(self):
        return hash(tuple(self.to_cluster()))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    @property
    def dimension(self):
        return len(vars(self).keys()) - len(self.cluster_exclude)

    @property
    def headers(self):
        return [key for key in vars(self).keys() if key not in self.cluster_exclude]

    def load(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.readonly:
                continue
            setattr(self, key, value)

    def __repr__(self):
        return str(vars(self))

    def to_cluster(self):
        res = []
        for key, value in vars(self).items():
            if key in self.cluster_exclude:
                continue
            if type(value) == np.ndarray:
                res.extend(value)
            else:
                res.append(value)
        return res

    def to_list(self):
        res = []
        for key, value in vars(self).items():
            if key in self.cluster_exclude:
                continue
            res.append(value)
        return res

    def from_list(self, lst):
        index = 0
        for key in vars(self).keys():
            if key in self.cluster_exclude:
                continue
            setattr(self, key, lst[index])
            index += 1

    def to_dict(self):
        """
        for json
        :return:
        """
        res = {}
        for key, value in vars(self).items():
            if key in self.readonly:
                continue
            res[key] = value
        return res


class State(Base):
    """
    状态类，包含状态信息
    """

    def __init__(self, **kwargs):
        super(State, self).__init__(**kwargs)

        self.state_type = None
        self.tag = None
        self.readonly = ['cluster_exclude', 'readonly', 'state_type', 'tag']
        # self.cluster_exclude = ['reward', 'is_crash', 'is_outoflane', 'is_reachdest']
        self.cluster_exclude = ['reward', 'is_outoflane', 'is_reachdest']
        self.cluster_exclude.extend(self.readonly)

    def load(self, **kwargs):
        all_zero = True
        all_one = True
        for key, value in kwargs.items():
            setattr(self, key, value)
            if type(value) == np.ndarray:
                all_zero &= (value == 0).all()
                all_one &= (value == 1).all()
            else:
                all_zero &= (value == 0)
                all_one &= (value == 1)

        if all_zero:
            self.state_type = 0
        elif all_one:
            self.state_type = 1


class Action(Base):
    def __init__(self, **kwargs):
        super(Action, self).__init__(**kwargs)

        self.readonly = ['cluster_exclude', 'readonly']
        self.cluster_exclude = []
        self.cluster_exclude.extend(self.readonly)


class ActionSpliter:
    def __init__(self, action_ranges, granularity):
        self.action_ranges = action_ranges
        self.granularity = granularity
        self._action_set = set()

        for key, value in self.action_ranges.items():
            self.action_ranges[key] = utils.expand_action_range(self.action_ranges[key], self.granularity[key])

        print(self.action_ranges)

    @property
    def action_len(self):
        size = 1
        for key, value in self.action_ranges.items():
            width = int((value[1] - value[0]) / self.granularity[key])
            size *= width
        return size

    def action2id(self, action: Action):
        res = 1  # 预留 action 0
        size = 1
        for key, value in vars(action).items():
            if key in action.readonly:
                continue
            width = int((self.action_ranges[key][1] - self.action_ranges[key][0]) / self.granularity[key])

            # offset = int(abs(self.action_ranges[key][0]) // self.granularity[key])
            tmp = int((value - self.action_ranges[key][0]) / self.granularity[key])
            if value == self.action_ranges[key][1]:
                tmp -= 1
            # res += (tmp + offset) * size
            res += tmp * size
            size *= width

        # self._action_set.add(res)
        # print(len(self._action_set)/size)
        # print(f'id: {res}')
        return res


class EnvType(enum.Enum):
    ACC = 'acc'
    LANE_KEEPING = 'lane_keeping'
    RACETRACK = 'racetrack'
    INTERSECTION = 'intersection'


class Trajectory:
    def __init__(self, config_data):
        self.config_data = config_data
        self.env_type = EnvType(self.config_data["env_type"])
        self.state = State()
        self.action = Action()

    def load(self, data):
        state_ranges = self.config_data['state_ranges']
        action_ranges = self.config_data['action_ranges']
        reward_ranges = self.config_data['reward_ranges']
        property_ranges = self.config_data['property_ranges']

        state = pd.Series(dtype='float64')
        for _range in state_ranges:
            state = pd.concat([state, data.iloc[_range[0]:_range[1]]])

        action = pd.Series(dtype='float64')
        for _range in action_ranges:
            action = pd.concat([action, data.iloc[_range[0]:_range[1]]])

        reward = pd.Series(dtype='float64')
        for _range in reward_ranges:
            reward = pd.concat([reward, data.iloc[_range[0]:_range[1]]])

        property = pd.Series(dtype='float64')
        for _range in property_ranges:
            property = pd.concat([property, data.iloc[_range[0]:_range[1]]])

        if self.env_type is EnvType.RACETRACK:
            state = state.to_numpy()
            self.state.load(on_road=state, **reward, **property)
        else:
            self.state.load(**state, **reward, **property)
        self.action.load(**action)


class Node:
    def __init__(self, state):
        self.state = state
        self.children = dict()  # (tag, action_id) -> weight

    def add_child(self, tag, action_id):
        if (tag, action_id) not in self.children:
            self.children[(tag, action_id)] = 1
        else:
            self.children[(tag, action_id)] += 1


class Graph:
    def __init__(self, data, K, action_spliter):
        self.data = data
        self.K = K
        self.action_spliter = action_spliter
        # tag -> state
        self.nodes = {}

    def gen_nodes(self):
        for state_tag in range(0, self.K + 1 + 1):
            state_tmp = []
            for j in range(len(self.data)):
                if self.data[j].state.tag == state_tag:
                    state_tmp.append(self.data[j].state)

            state_mean = mean(state_tmp)
            if state_mean is None:
                continue
            self.nodes[state_tag] = Node(state_mean)

    def gen_edges(self):
        for key in self.nodes:
            node = self.nodes[key]
            for j in range(len(self.data) - 1):
                if self.data[j].state.tag != node.state.tag:
                    continue

                # next_episode
                if node.state.tag == self.K + 1:
                    # print(f'special state {node.state.tag}')
                    continue

                # remove loop
                # if node.state.tag == self.data[j + 1].state.tag:
                #     continue
                node.add_child(self.data[j + 1].state.tag, self.action_spliter.action2id(self.data[j].action))

                # if state_tag != 0 and state_tag != self.K + 1:
                #     tag_mark.add(state_tag)


class Property:
    def __init__(self, name):
        self.name = name

    def get_value(self, result_list, index):
        return float(result_list[index])


class EpisodeRewardProperty(Property):
    def __init__(self, name='episode_reward'):
        super(EpisodeRewardProperty, self).__init__(name)

    def get_value(self, result_list, index):
        return float(result_list[index])


class OppositeProperty(Property):
    def __init__(self, name):
        super(OppositeProperty, self).__init__(name)

    def get_value(self, result_list, index):
        return 1 - float(result_list[index])
