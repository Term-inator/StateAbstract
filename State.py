import numpy as np


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

        self.readonly = ['readonly']

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
            if key in self.readonly:
                continue
            res.append(value)
        return res

    def from_list(self, lst):
        for key, value in zip(vars(self).keys(), lst):
            if key in self.readonly:
                continue
            setattr(self, key, value)

    def to_dict(self):
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
        self.readonly = ['readonly', 'state_type', 'tag']

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

        self.readonly = ['readonly']


class ActionSpliter:
    def __init__(self, action_range=None, granularity=None):
        if action_range is None:
            self.action_range = {'acc': [-3.0, 3.0], 'steer': [-0.3, 0.3]}
        if granularity is None:
            self.granularity = {'acc': 0.1, 'steer': 0.01}

    def split(self, action: Action):
        res = {}
        for key, value in vars(action).items():
            if key in action.readonly:
                continue
            if key in self.granularity:
                res[key] = int(value // self.granularity[key])
                if value == self.action_range[key][1]:
                    res[key] -= 1
            else:
                raise ValueError(f'key {key} not in granularity')
        return Action(**res)

    def action2id(self, action: Action):
        # dimension = len(vars(action).items()) - len(action.readonly)
        res = 0
        for key, value in vars(action).items():
            if key in action.readonly:
                continue
            width = int((self.action_range[key][1] - self.action_range[key][0]) / self.granularity[key])
            offset = int(abs(self.action_range[key][0]) / self.granularity[key])
            res += (value + offset) * width
        return res


class Trajectory:
    def __init__(self):
        self.state = State()
        self.action = Action()

    def load(self, data):
        self.state.load(**data.iloc[0:2])
        self.action.load(**data.iloc[2:3])


class Node:
    def __init__(self, state):
        self.state = state
        self.children = dict()  # (tag, action) -> weight

    def add_child(self, tag, action):
        if (tag, action) not in self.children:
            self.children[(tag, action)] = 1
        else:
            self.children[(tag, action)] += 1


class Graph:
    def __init__(self, data, K):
        self.data = data
        self.K = K
        self.action_spliter = ActionSpliter()
        # tag -> state
        self.nodes = {}

        self.gen_nodes()

    def gen_nodes(self):
        for state_tag in range(0, self.K + 1 + 1):
            state_tmp = []
            for j in range(len(self.data)):
                if self.data[j].state.tag == state_tag:
                    state_tmp.append(self.data[j].state)

            state_mean = mean(state_tmp)
            if state_mean is None:
                continue
            # print(state_tag, state_mean)
            self.nodes[state_tag] = Node(state_mean)

    def gen(self):
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
                if node.state.tag == self.data[j + 1].state.tag:
                    continue
                node.add_child(self.data[j + 1].state.tag, self.action_spliter.split(self.data[j].action))

                # if state_tag != 0 and state_tag != self.K + 1:
                #     tag_mark.add(state_tag)
