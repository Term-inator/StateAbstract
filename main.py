import queue
import threading

import joblib
from tqdm import tqdm

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
    'env': '../RL-Carla/output_logger/env-lane-icm dnn-dest150m-after0225-reward500/trajectory_2000.csv',
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


def draw_graph(graph, K):
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


def SSE(data, graph):
    # 计算 SSE
    sse = 0
    for i in range(len(data)):
        sse += np.linalg.norm(np.array(data[i].state.to_list()) - graph.nodes[data[i].state.tag].state.to_list()) ** 2
    return sse


def check_steady(data, K, slide_window=5):
    steady = 0
    total_n = 0
    tags = []
    for i in range(len(data)):
        if data[i].state.tag == 0 or data[i].state.tag == K + 1:
            tags = []
            continue
        tags.append(data[i].state.tag)
        # print(tags)
        if len(tags) == slide_window:
            dif = len(set(tags))
            # steady += (slide_window - dif) / slide_window
            steady += utils.calc_prob(K, slide_window, dif)
            total_n += 1
            tags.pop(0)
    return steady / total_n


# 用于存储模型，避免重复计算
model_memo = {}  # K -> {model, graph, times} times为被使用的次数，达到 epoch 时清除


def get_model_and_graph(epoch, K, data, states, data_type='env', slide_window=5, parallel=False):
    if K in model_memo:
        if model_memo[K]['times'] >= epoch:
            model_memo.pop(K)
        else:
            model_memo[K] = {
                'model': model_memo[K].model,
                'graph': model_memo[K].graph,
                'times': model_memo[K].times + 1
            }
    else:
        model = _cluster(K, states)
        set_label(data, model, K, 'env')
        graph = Graph(data, K)
        model_memo[K] = {
            'model': model,
            'graph': graph,
            'times': 1
        }
    return model_memo[K]['model'], model_memo[K]['graph']


def try_K(epoch, K_min, K_max, data, states, data_type='env', slide_window=5, parallel=False):
    sse_scores = []
    sc_scores = []
    ch_scores = []
    steady_scores = []

    if parallel:
        def calc_sse(data, graph, sse_scores):
            # print('calc_sse')
            sse_scores.append(SSE(data, graph))

        def calc_sc(states, labels, sc_scores):
            # print('calc_sc')
            sc_scores.append(
                sklearn.metrics.silhouette_score(states.data, labels, metric='euclidean', sample_size=1000))

        def calc_ch(states, labels, ch_scores):
            # print('calc_ch')
            ch_scores.append(sklearn.metrics.calinski_harabasz_score(states.data, labels))

        def calc_steady(data, K, slide_window, steady_scores):
            # print('calc_steady')
            steady_scores.append(check_steady(data, K, slide_window))

        # global model_memo
        for K in range(K_min, K_max):
            print(f"Processing epoch {epoch}, K {K}")
            model, graph = get_model_and_graph(epoch, K, data, states, data_type, slide_window, parallel)

            sse_thread = threading.Thread(target=calc_sse, args=(data, graph, sse_scores))
            sse_thread.start()

            sc_thread = threading.Thread(target=calc_sc, args=(states, model.labels_, sc_scores))
            sc_thread.start()

            ch_thread = threading.Thread(target=calc_ch, args=(states, model.labels_, ch_scores))
            ch_thread.start()

            steady_thread = threading.Thread(target=calc_steady, args=(data, K, slide_window, steady_scores))
            steady_thread.start()

            sse_thread.join()
            sc_thread.join()
            ch_thread.join()
            steady_thread.join()
    else:
        for K in range(K_min, K_max):
            model, graph = get_model_and_graph(epoch, K, data, states, data_type, slide_window, parallel)
            sse_score = SSE(data, graph)
            sc_score = sklearn.metrics.silhouette_score(states.data, model.labels_, metric='euclidean', sample_size=1000)
            ch_score = sklearn.metrics.calinski_harabasz_score(states.data, model.labels_)
            steady_score = check_steady(data, K, slide_window)

            sse_scores.append(sse_score)
            sc_scores.append(sc_score)
            ch_scores.append(ch_score)
            steady_scores.append(steady_score)

    return sse_scores, sc_scores, ch_scores, steady_scores


def monte_carlo(data, states, data_type='env', epochs=20, parallel=False):
    K_min = 10
    K_max = 60
    matx_sse = np.mat(np.zeros((epochs, K_max - K_min)))
    matx_sc = np.mat(np.zeros((epochs, K_max - K_min)))
    matx_ch = np.mat(np.zeros((epochs, K_max - K_min)))
    matx_st = np.mat(np.zeros((epochs, K_max - K_min)))
    slide_window = 5
    utils.init_memo(K_max, slide_window)

    if parallel:
        # 创建任务队列
        task_queue = queue.Queue()
        for i in range(epochs):
            task_queue.put(i)

        # 定义线程函数
        def worker():
            while True:
                try:
                    # 从任务队列中取出一个任务
                    i = task_queue.get(block=False)
                except queue.Empty:
                    # 任务队列为空，退出线程
                    return

                # print(f'epoch {i}')
                sse_scores, sc_scores, ch_scores, steady_scores = try_K(i, K_min, K_max, data, states,
                                                                        data_type=data_type,
                                                                        slide_window=slide_window,
                                                                        parallel=True)
                matx_sse[i, :] = sse_scores
                matx_sc[i, :] = sc_scores
                matx_ch[i, :] = ch_scores
                matx_st[i, :] = steady_scores

        # 创建n个线程并运行
        n_threads = epochs
        threads = []
        for i in range(n_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()
    else:
        for i in range(epochs):
            sse_scores, sc_scores, ch_scores, steady_scores = try_K(i, K_min, K_max, data, states,
                                                                    data_type=data_type,
                                                                    slide_window=slide_window,
                                                                    parallel=False)
            matx_sse[i, :] = sse_scores
            matx_sc[i, :] = sc_scores
            matx_ch[i, :] = ch_scores
            matx_st[i, :] = steady_scores

    mean_sse = matx_sse.sum(axis=0) / epochs
    mean_sc = matx_sc.sum(axis=0) / epochs
    mean_ch = matx_ch.sum(axis=0) / epochs
    mean_ch_norm = mean_ch / max(mean_ch.tolist()[0]) / 3
    mean_st = matx_st.sum(axis=0) / epochs

    fig1 = plt.figure(figsize=(15, 8))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    X = range(K_min, K_max)
    ax1.plot(X, mean_sse.tolist()[0], marker='o', label='SSE')
    ax2.plot(X, mean_sc.tolist()[0], 'r', marker='*', label='Silhouette')
    ax2.plot(X, mean_ch_norm.tolist()[0], 'g', marker='*', label='Calinski Harabasz')

    ax1.set_ylabel('SSE', fontsize=20)
    ax1.set_xlabel('K', fontsize=20)
    ax2.set_ylabel('Value', fontsize=20)
    ax1.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    ax1.legend(loc='lower left', fontsize=20)
    ax2.legend(loc='upper right', fontsize=20)
    fig1.savefig(f'./{data_type}_sse_sc_ch.png')

    fig2 = plt.figure(figsize=(15, 8))
    ax3 = fig2.add_subplot(1, 1, 1)
    ax3.plot(X, mean_st.tolist()[0], 'b', marker='o', label='Steady')
    ax3.legend(loc='upper right', fontsize=20)
    ax3.set_xlabel('K', fontsize=20)
    ax3.set_ylabel('Value', fontsize=20)
    fig2.savefig(f'./{data_type}_steady.png')


if __name__ == '__main__':
    # test K
    env_data, env_states = load_data(params['env'])
    monte_carlo(env_data, env_states, data_type='env', epochs=20, parallel=True)

    # K = 15
    # env_data, env_states = load_data(params['env'])
    # model = _cluster(K, env_states)
    # set_label(env_data, model, K, 'env')
    # env_graph = Graph(env_data, K)
    # env_graph.gen()

    # policy_data, policy_states = load_data(params['policy'])
    # set_label(policy_data, model, K, 'policy')
    # policy_graph = Graph(policy_data, K)
    # policy_graph.gen()

    # draw_graph(policy_graph)
    # draw_heatmap(env_data)
    # _parser = prism_parser.PrismParser(K, env_graph, policy_graph)
    # code = _parser.parse(save=True)
    # print(code)
    # _parser = uppaal_parser.UppaalParser(graph)
    # xml_tree = _parser.to_xml()
    # uppaal_parser.write(xml_tree, 'uppaal')
    pass
