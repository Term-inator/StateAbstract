"""
力引导图布局
"""
import random
import math
from matplotlib import pyplot as plt
import networkx as nx
import time
import numpy as np


class ForceDirectedLayout:
    # 模型参数
    K_r = 6
    K_s = 0.3
    edge_length = 5
    delta_t = 50
    max_length = 30
    # color = ['red', 'green', 'blue', 'orange']
    node_num: int
    original_max_posx = 400
    original_max_posy = 400
    # 图形容器
    node_force = {}
    node_position = {}

    # 与采样退火有关的参数
    displacement_list = []  # 用于采样的列表
    scale = 3  # 采样的范围

    def __init__(self, graph):
        self.graph = graph
        self.node_num = len(self.graph.nodes)

        # 初始化节点位置
        for i in range(0, self.node_num):
            self.node_position[i] = np.array([random.random() * self.original_max_posx, random.random() * self.original_max_posy])
            self.node_force[i] = np.array([0, 0])

    def compute_repulsion(self):  # 计算每两个点之间的斥力
        for i in range(0, self.node_num):
            for j in range(i + 1, self.node_num):
                distance_square = np.square(self.node_position[i] - self.node_position[j]).sum()
                if distance_square != 0:
                    distance = math.sqrt(distance_square)

                    state_x = np.array(self.graph.nodes[i].state.to_cluster())
                    state_y = np.array(self.graph.nodes[j].state.to_cluster())

                    # difference = np.square(state_x - state_y).sum()
                    difference = np.linalg.norm(state_x - state_y)
                    force = self.K_r * difference / distance_square

                    f = force * (self.node_position[j] - self.node_position[i]) / distance

                    self.node_force[i] = self.node_force[i] - f
                    self.node_force[j] = self.node_force[j] + f

    def compute_string(self):
        for i in range(0, self.node_num):
            for j, action in self.graph.nodes[i].children:
                if i < j:
                    distance = np.linalg.norm(np.array(self.node_position[i] - self.node_position[j]))
                    if distance != 0:
                        force = self.K_s * (distance - self.edge_length)

                        f = force * (self.node_position[j] - self.node_position[i]) / distance

                        self.node_force[i] = self.node_force[i] + f
                        self.node_force[j] = self.node_force[j] - f

    def update_position(self, times):  # 更新坐标
        displacement_sum = 0
        for i in range(0, self.node_num):
            displacement = self.node_force[i] * self.delta_t
            displacement_len_square = np.square(displacement).sum()
            displacement_len = np.sqrt(displacement_len_square)

            # 随迭代次数增加，MaxLength逐渐减小；
            # current_max_length = self.max_length / (times + 0.1)
            current_max_length = self.max_length

            if displacement_len_square > current_max_length:
                s = math.sqrt(current_max_length / displacement_len_square)
                displacement = displacement * s
            (new_x, new_y) = self.node_position[i] + displacement
            displacement_sum += displacement_len
            self.node_position[i] = np.array([new_x, new_y])
        return displacement_sum

    def run(self, iterations=200):
        start = time.perf_counter()
        iteration_time = 0
        for times in range(0, 1 + iterations):
            for i in range(0, self.node_num):
                self.node_force[i] = np.array([0, 0])
            self.compute_repulsion()
            self.compute_string()
            # 记录本次迭代移动距离：
            displacement_sum = self.update_position(times)
            self.displacement_list.append(displacement_sum)
            print(displacement_sum)
            # if len(self.displacement_list) > self.scale:
            #     last = np.mean(self.displacement_list[times - self.scale - 1:times - 1])
            #     now = np.mean(self.displacement_list[times - self.scale:times])
            #     if (last - now) / last < 0.01:
            #         break
            iteration_time = times
        end = time.perf_counter()

        print('Running time: %s Seconds' % (end - start))
        print('最终迭代次数:', iteration_time)

        return self.node_position
