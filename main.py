import joblib

import force_directed_layout
import prism_parser

import numpy as np
import pandas as pd
import sklearn
import sklearn.datasets as datasets
import sklearn.cluster as cluster
from sklearn.utils import Bunch
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

import uppaal_parser
import utils
from State import State, Trajectory, Graph
from utils import read_csv

matplotlib.use('TkAgg')


def load_states(data):
    lst = []
    for i in range(len(data)):
        # 开始结状态
        if data[i].state.state_type != '':
            continue
        lst.append(data[i].state.to_list())
    lst = np.array(lst, dtype='float32')

    return Bunch(
        data=lst,
        # target=target,
        # frame=frame,
        # target_names=target_names,
        # DESCR=fdescr,
        # feature_names=feature_names,
        # filename=data_file_name,
        # data_module=DATA_MODULE,
    )


params = {
    'env': '../RL-Carla/output_logger/env-lane-icm dnn-dest150m-after0225-reward500/trajectory_5000.csv',
    'policy': 'trajectory_vit_500.csv',
}


def load_data(filename):
    data = read_csv(filename)
    states = load_states(data)
    return data, states


def _cluster(K, states):
    # model = cluster.AgglomerativeClustering(K)
    model = cluster.Birch(n_clusters=K)
    # model = cluster.MiniBatchKMeans(K)
    model.fit(states.data)
    # print(model.cluster_centers_)
    # print(model.labels_)
    joblib.dump(model, f'cluster.pkl')
    return model


def set_label(data, model, K, type):
    label_index = 0
    for i in range(len(data)):
        if data[i].state.state_type == 'start':
            data[i].state.tag = 0
        elif data[i].state.state_type == 'end':
            data[i].state.tag = K + 1
        else:
            if type == 'env':
                data[i].state.tag = model.labels_[label_index] + 1
                label_index += 1
            else:
                data[i].state.tag = model.predict([data[i].state.to_list()])[0] + 1


def draw_graph(graph):
    G = nx.DiGraph()
    G.clear()
    for key in graph.nodes:
        node = graph.nodes[key]
        if node is None:
            continue
        G.add_node(node.state.tag, desc=node.state.speed)
        # print(node.state.tag, len(node.children))
        for child in node.children:
            next_state_tag = child[0]
            action = child[1]
            G.add_edge(node.state.tag, next_state_tag, name=f'{action[0]} {action[1]} {node.children[child]}')

    layout = force_directed_layout.ForceDirectedLayout(graph)
    pos = layout.run(iterations=200)

    color_map = []
    for node in G:
        color = '#1f78b4'
        if node == 0 or node == K + 1:
            color = 'green'
        else:
            if graph.nodes[node].state.cost is not None and graph.nodes[node].state.cost > 50:
                color = 'red'
        color_map.append(color)

    fig = plt.figure()
    # pos = nx.spring_layout(G, scale=2)
    nx.draw(G, pos, node_color=color_map, with_labels=True, font_size=8)
    # node_labels = nx.get_node_attributes(G, 'desc')
    # edge_labels = nx.get_edge_attributes(G, 'name')
    # nx.draw_networkx_labels(G, pos, labels=node_labels)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # plt.show()
    fig.savefig('graph.png')


def draw_heatmap(data):
    # 获取 x, y 范围
    min_x, max_x = 100000, -100000
    min_y, max_y = 100000, -100000
    for i in range(len(data)):
        # 开始结束状态
        if data[i].state.state_type != '':
            continue
        min_x = min(min_x, data[i].state.coordinate[0])
        max_x = max(max_x, data[i].state.coordinate[0])
        min_y = min(min_y, data[i].state.coordinate[1])
        max_y = max(max_y, data[i].state.coordinate[1])
    print(min_x, max_x, min_y, max_y)

    x_span = 0.5
    y_span = 0.5
    min_x = min_x // x_span * x_span
    max_x = (max_x // x_span + 1) * x_span
    min_y = min_y // y_span * y_span
    max_y = (max_y // y_span + 1) * y_span
    x_range = max_x - min_x
    y_range = max_y - min_y

    # 长宽不一致，补齐
    if x_range > y_range:
        max_y += + np.floor((x_range - y_range) / 2)
        min_y -= np.floor((x_range - y_range) / 2)
    else:
        max_x += np.floor((y_range - x_range) / 2)
        min_x -= np.floor((y_range - x_range) / 2)

    x = np.arange(min_x, max_x + x_span, x_span)
    y = np.arange(min_y, max_y + y_span, y_span)
    x = np.apply_along_axis(lambda x: np.round(x), 0, x)
    y = np.apply_along_axis(lambda x: np.round(x), 0, y)

    state_mark_set = [[set() for j in range(len(y))] for i in range(len(x))]
    map_data = np.zeros((len(x), len(y)))

    for i in range(len(data)):
        # 开始结束状态
        if data[i].state.state_type != '':
            continue
        x_index = int((data[i].state.coordinate[0] - min_x) // x_span)
        y_index = int((data[i].state.coordinate[1] - min_y) // y_span)
        state_mark_set[x_index][y_index].add(data[i].state.tag)
        # map_data[x_index][y_index] += 1

    for i in range(len(x)):
        for j in range(len(y)):
            map_data[i][j] = len(state_mark_set[i][j])

    # map_data = utils.normalization(map_data)

    map_data_frame = pd.DataFrame(
        data=map_data,
        columns=y,
        index=x
    )
    print(map_data_frame)

    fig = plt.figure()
    heatmap = sns.heatmap(map_data_frame)
    # plt.show()
    fig.savefig(f'heatmap.png')


# def get_cluster_center(data):
#     # 获取聚类中心
#     cluster_center = []
#     for i in range(len(model.cluster_centers_)):
#         cluster_center.append(model.cluster_centers_[i])
#     return cluster_center
#
#
# def SSE():
#     # 计算 SSE
#     sse = 0
#     for i in range(len(data)):
#         sse += np.linalg.norm(data[i].state.coordinate - model.cluster_centers_[data[i].label])
#     return sse


if __name__ == '__main__':
    K = 15
    env_data, env_states = load_data(params['env'])
    model = _cluster(K, env_states)
    set_label(env_data, model, K, 'env')
    env_graph = Graph(env_data, K)
    env_graph.gen()

    # policy_data, policy_states = load_data(params['policy'])
    # set_label(policy_data, model, K, 'policy')
    # policy_graph = Graph(policy_data, K)
    # policy_graph.gen()

    # draw_graph(policy_graph)
    draw_heatmap(env_data)
    # _parser = prism_parser.PrismParser(K, env_graph, policy_graph)
    # code = _parser.parse(save=True)
    # print(code)
    # _parser = uppaal_parser.UppaalParser(graph)
    # xml_tree = _parser.to_xml()
    # uppaal_parser.write(xml_tree, 'uppaal')
    pass
