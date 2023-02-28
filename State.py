import numpy as np


def mean(lst):
    if len(lst) == 0:
        return None
    elif len(lst) == 1:
        return lst[0]
    else:
        return lst[0].mean(lst[1:])


e_speed = 10
e_distance = 20
safe_distance = 10


class State:
    """
    状态类，包含状态信息
    """

    def __init__(self):
        self.velocity_t_x = None
        self.velocity_t_y = None
        self.accel_t_x = None
        self.accel_t_y = None
        self.delta_yaw_t = None
        self.dyaw_dt_t = None
        self.lateral_dist_t = None
        self.action_last_accel = None
        self.accel_last_steer = None
        self.future_angles_0 = None
        self.future_angles_1 = None
        self.future_angles_3 = None

        self.speed = 0.0
        self.angle = 0.0
        self.offset = 0.0
        self.e_speed = e_speed
        self.distance = -1
        self.e_distance = e_distance
        self.safe_distance = safe_distance
        self.light_state = None
        self.sl_distance = 50
        self.cost = None

        self.tag = 0
        self.state_type = ''
        self.coordinate = None

    def load(self, data):
        """
        ['velocity_t_x', 'velocity_t_y', 'accel_t_x', 'accel_t_y',
         'delta_yaw_t', 'dyaw_dt_t', 'lateral_dist_t',
         'action_last_accel', 'accel_last_steer',
         'future_angles_0', 'future_angles_1', 'future_angles_3',
         'speed', 'angle', 'offset', 'e_speed', 'distance',
         'e_distance', 'safe_distance', 'light_state', 'sl_distance',

         'action_acc', 'action_steer', 'x', 'y', 'cost']
        """
        self.speed = data['speed']
        self.angle = data['angle']
        self.offset = data['offset']
        # self.e_speed = data['e_speed']
        # self.distance = data['distance']
        # self.e_distance = data['e_distance']
        # self.safe_distance = data['safe_distance']
        # self.light_state = data['light_state']
        # self.sl_distance = data['sl_distance']

        self.coordinate = (data['x'], data['y'])

        if 'cost' in data:
            self.cost = data['cost']

    def __repr__(self):
        if self.cost is None:
            return f'({self.speed}, {self.angle}, {self.offset}, {self.e_speed}, {self.distance}, ' \
                   f'{self.e_distance}, {self.safe_distance}, {self.light_state}, {self.sl_distance})'
        return f'({self.speed}, {self.angle}, {self.offset}, {self.e_speed}, {self.distance}, ' \
               f'{self.e_distance}, {self.safe_distance}, {self.light_state}, {self.sl_distance}, {self.cost})'

    def to_list(self):
        if self.cost is None:
            return [self.speed, self.angle, self.offset]
        return [self.speed, self.angle, self.offset, self.cost]

    def to_dict(self):
        return {
            'velocity_t_x': self.velocity_t_x,
            'velocity_t_y': self.velocity_t_y,
            'accel_t_x': self.accel_t_x,
            'accel_t_y': self.accel_t_y,
            'delta_yaw_t': self.delta_yaw_t,
            'dyaw_dt_t': self.dyaw_dt_t,
            'lateral_dist_t': self.lateral_dist_t,
            'action_last_accel': self.action_last_accel,
            'accel_last_steer': self.accel_last_steer,
            'future_angles_0': self.future_angles_0,
            'future_angles_1': self.future_angles_1,
            'future_angles_3': self.future_angles_3,

            'speed': self.speed,
            'angle': self.angle,
            'offset': self.offset,
            # 'e_speed': self.e_speed,
            # 'distance': self.distance,
            # 'e_distance': self.e_distance,
            # 'safe_distance': self.safe_distance,
            # 'light_state': self.light_state,
            # 'sl_distance': self.sl_distance,
        }

    def add(self, state):
        self.speed += state.speed
        self.angle += state.angle
        self.offset += state.offset
        # self.e_speed += state.e_speed
        # self.distance += state.distance
        # self.e_distance += state.e_distance
        # self.safe_distance += state.safe_distance
        # self.light_state = np.random.choice([self.light_state, state.light_state], 1)[0]
        # self.sl_distance += state.sl_distance
        if self.cost is not None:
            self.cost += state.cost

    def divide(self, n):
        self.speed /= n
        self.angle /= n
        self.offset /= n
        # self.e_speed /= n
        # self.distance /= n
        # self.e_distance /= n
        # self.safe_distance /= n
        # self.sl_distance /= n
        if self.cost is not None:
            self.cost /= n

    def mean(self, states):
        state_mean = State()
        for state in states:
            state_mean.add(state)
        state_mean.divide(len(states))
        state_mean.tag = states[0].tag
        return state_mean


class Trajectory:
    def __init__(self):
        self.state = State()
        self.action = [0, 0]

    def load(self, data):
        self.state.load(data)
        self.action[0] = data['action_acc']
        self.action[1] = data['action_steer']

        if self.state.speed == 0:
            if self.action[0] == 0:
                self.state.state_type = 'start'
            else:
                self.state.state_type = 'end'


class ActionSpliter:
    def __init__(self, action_acc_range=None, action_steer_range=None, acc_split=0.5, steer_split=0.05):
        if action_steer_range is None:
            action_steer_range = [-0.3, 0.3]
        if action_acc_range is None:
            action_acc_range = [-3.0, 3.0]
        self.action_acc_range = action_acc_range
        self.action_steer_range = action_steer_range
        self.acc_split = acc_split
        self.steer_split = steer_split

    def split(self, action):
        return action[0] // self.acc_split, action[1] // self.steer_split


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
            print(state_tag, state_mean)
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
