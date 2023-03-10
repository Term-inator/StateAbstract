import json

import numpy as np
import pandas as pd

import utils


def read_csv(filename):
    data_csv = pd.read_csv(filename)

    # data_csv = utils.delete_out_three_sigma(data_csv)
    # data_csv = delete_threshold(data_csv, threshold=20)

    data = []
    for i in data_csv.index:
        trajectory = Trajectory()
        # print(data_csv.loc[i].values[0:-1])
        trajectory.load(data_csv.loc[i])
        # print(state.__repr__())
        data.append(trajectory)

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

        self.readonly = ['cluster_exclude', 'readonly']
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
        return hash(tuple(self.to_list()))

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

    def to_list(self):
        res = []
        for key, value in vars(self).items():
            if key in self.cluster_exclude:
                continue
            res.append(value)
        return res

    def from_list(self, lst):
        for key, value in zip(vars(self).keys(), lst):
            if key in self.cluster_exclude:
                continue
            setattr(self, key, value)

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
    ??????????????????????????????
    """

    def __init__(self, **kwargs):
        super(State, self).__init__(**kwargs)

        self.state_type = None
        self.tag = None
        self.readonly = ['cluster_exclude', 'readonly', 'state_type', 'tag']
        self.cluster_exclude = ['reward']
        self.cluster_exclude.extend(self.readonly)

    def load(self, **kwargs):
        all_zero = True
        all_one = True
        for key, value in kwargs.items():
            setattr(self, key, value)
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

        for key, value in self.action_ranges.items():
            self.action_ranges[key] = utils.expand_action_range(self.action_ranges[key], self.granularity[key])

        print(self.action_ranges)

    @property
    def action_len(self):
        size = 1
        for key, value in self.action_ranges.items():
            width = int((value[1] - value[0]) // self.granularity[key])
            size *= width
        return size

    def action2id(self, action: Action):
        res = 1  # ?????? action 0
        size = 1
        for key, value in vars(action).items():
            if key in action.readonly:
                continue
            width = int((self.action_ranges[key][1] - self.action_ranges[key][0]) // self.granularity[key])
            offset = int(abs(self.action_ranges[key][0]) // self.granularity[key])
            tmp = int(value // self.granularity[key])
            if tmp == self.action_ranges[key][1]:
                tmp -= 1
            res += (tmp + offset) * size
            size *= width
        # print(f'id: {res}')
        return res


class Trajectory:
    def __init__(self):
        self.state = State()
        self.action = Action()

    def load(self, data):
        self.state.load(**data.iloc[0:2], **data.iloc[3:4])
        self.action.load(**data.iloc[2:3])


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
