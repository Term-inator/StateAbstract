import copy
import datetime
import json
import os
import queue
import re
import subprocess
import threading
import time
from time import sleep

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import sklearn.cluster as cluster
from sklearn.utils import Bunch

import prism_parser
import utils
from base import Graph, read_csv, ActionSpliter, EnvType

matplotlib.use('TkAgg')


def load_states(data):
    lst = []
    for i in range(len(data)):
        # 开始结束状态
        if data[i].state.state_type is not None:
            continue
        lst.append(data[i].state.to_cluster())
    lst = np.array(lst, dtype='float32')

    return Bunch(
        data=lst,
        # target=target,
        # frame=frame,
        # target_names=target_names,
        # DESCR=descr,
        # feature_names=feature_names,
        # filename=data_file_name,
        # data_module=DATA_MODULE,
    )


config = utils.load_yml('./configs/acc.yaml')
prism_path = 'C:/Program Files/prism-4.7/bin'


def load_data(config_data, data_type: str):
    data = read_csv(config_data, data_type)

    if data_type == 'env':
        _data, _mean, _std = utils.standardize(copy.deepcopy(data))
        config['data']['env']['mean'] = _mean
        config['data']['env']['std'] = _std
    elif data_type == 'policy':
        _data, _, _ = utils.standardize(copy.deepcopy(data),
                                        _mean=config['data']['env']['mean'],
                                        _std=config['data']['env']['std'])
    else:
        raise Exception('data type error')
    _states = load_states(data)
    return data, _data, _states


def _cluster(K, states, cluster_type):
    """
    1. kmeans
    2. mini batch kmeans
    3. birch
    1.2. 基于划分
    3. 基于层次
    """
    print(f'cluster type: {cluster_type}, K: {K}')
    if cluster_type == 'kmeans':
        model = cluster.KMeans(K)
    elif cluster_type == 'mini_batch_kmeans':
        model = cluster.MiniBatchKMeans(K)
    elif cluster_type == 'agglomerative':
        model = cluster.AgglomerativeClustering(n_clusters=K)
    elif cluster_type == 'birch':
        model = cluster.Birch(n_clusters=K, threshold=config['cluster']['threshold'], compute_labels=True)
    else:
        raise Exception('cluster type error')
    model.fit(states.data)
    # print(model.cluster_centers_)
    # print(model.labels_)
    # joblib.dump(model, f'cluster.pkl')
    return model


def _predict(model, state, cluster_type):
    if cluster_type == 'kmeans':
        return model.predict(state)
    elif cluster_type == 'mini_batch_kmeans':
        return model.predict(state)
    elif cluster_type == 'agglomerative':
        return model.fit_predict(state)
    elif cluster_type == 'birch':
        return model.predict(state)
    else:
        raise Exception('cluster type error')


def set_label(data, model, K, cluster_type, type):
    label_index = 0
    for i in range(len(data)):
        if data[i].state.state_type == 0:
            data[i].state.tag = 0
        elif data[i].state.state_type == 1:
            data[i].state.tag = K + 1
        else:
            if type == 'env':
                data[i].state.tag = model.labels_[label_index] + 1
                label_index += 1
            else:
                data[i].state.tag = _predict(model, np.array([data[i].state.to_cluster()], dtype='float32'), cluster_type)[
                                        0] + 1


def draw_graph(graph, K):
    G = nx.DiGraph()
    G.clear()
    for key in graph.nodes:
        node = graph.nodes[key]
        if node is None:
            continue
        G.add_node(node.state.tag)
        # print(node.state.tag, len(node.children))
        for child in node.children:
            next_state_tag = child[0]
            action = child[1]
            G.add_edge(node.state.tag, next_state_tag)

    # layout = force_directed_layout.ForceDirectedLayout(graph)
    # pos = layout.run(iterations=200)

    color_map = []
    for node in G:
        color = '#1f78b4'
        if node == 0 or node == K + 1:
            color = 'green'
        # else:
        #     if graph.nodes[node].state.cost is not None and graph.nodes[node].state.cost > 50:
        #         color = 'red'
        color_map.append(color)

    fig = plt.figure()
    pos = nx.spring_layout(G, scale=2)
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
        if data[i].state.state_type is not None:
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
        if data[i].state.state_type is not None:
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
        sse += np.linalg.norm(np.array(data[i].state.to_cluster()) - graph.nodes[data[i].state.tag].state.to_cluster()) ** 2
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


# TODO 作为 try_K 参数
# 用于存储模型，避免重复计算
model_memo = {}  # (K, cluster_type) -> {model, graph, times} times为被使用的次数，达到 epoch 时清除


def get_model_and_graph(epochs, K, data, states, action_spliter, data_type='env', cluster_type='birch', slide_window=5,
                        parallel=False):
    if K in model_memo:
        if model_memo[(K, cluster_type)]['times'] > epochs:
            print(f'K={K}, {cluster_type} model has been used {model_memo[K]["times"]} times, clear it now.')
            dt = model_memo.pop((K, cluster_type))
            return dt['model'], dt['graph']
        else:
            model_memo[(K, cluster_type)]['times'] += 1
            return model_memo[(K, cluster_type)]['model'], model_memo[K]['graph']
    else:
        model = _cluster(K, states, cluster_type)
        set_label(data, model, K, cluster_type, data_type)
        graph = Graph(data, K, action_spliter=action_spliter)
        graph.gen_nodes()
        model_memo[(K, cluster_type)] = {
            'model': model,
            'graph': graph,
            'times': 1
        }
        return model_memo[(K, cluster_type)]['model'], model_memo[(K, cluster_type)]['graph']


def try_K(epochs, current_epoch, K_min, K_max, data, states, action_spliter, data_type='env', cluster_type='birch',
          slide_window=5, calc_type=0b1111, parallel=False):
    sse_scores = []
    sc_scores = []
    ch_scores = []
    st_scores = []

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
            print(f"Processing epoch {current_epoch}, K {K}")
            # TODO 移到循环外？
            model_read_write_lock = threading.Lock()
            with model_read_write_lock:
                model, graph = get_model_and_graph(epochs=epochs, K=K, data=data, states=states,
                                                   action_spliter=action_spliter,
                                                   data_type=data_type, cluster_type=cluster_type,
                                                   slide_window=slide_window, parallel=parallel)

            if calc_type & 0b1000:
                sse_thread = threading.Thread(target=calc_sse, args=(data, graph, sse_scores))
                sse_thread.start()

            if calc_type & 0b0100:
                sc_thread = threading.Thread(target=calc_sc, args=(states, model.labels_, sc_scores))
                sc_thread.start()

            if calc_type & 0b0010:
                ch_thread = threading.Thread(target=calc_ch, args=(states, model.labels_, ch_scores))
                ch_thread.start()

            if calc_type & 0b0001:
                st_thread = threading.Thread(target=calc_steady, args=(data, K, slide_window, st_scores))
                st_thread.start()

            if calc_type & 0b1000:
                sse_thread.join()
            if calc_type & 0b0100:
                sc_thread.join()
            if calc_type & 0b0010:
                ch_thread.join()
            if calc_type & 0b0001:
                st_thread.join()
    else:
        for K in range(K_min, K_max):
            model, graph = get_model_and_graph(epochs=epochs, K=K, data=data, states=states,
                                               action_spliter=action_spliter,
                                               data_type=data_type, cluster_type=cluster_type,
                                               slide_window=slide_window, parallel=parallel)
            if calc_type & 0b1000:
                sse_score = SSE(data, graph)
                sse_scores.append(sse_score)

            if calc_type & 0b0100:
                sc_score = sklearn.metrics.silhouette_score(states.data, model.labels_, metric='euclidean',
                                                            sample_size=1000)
                sc_scores.append(sc_score)

            if calc_type & 0b0010:
                ch_score = sklearn.metrics.calinski_harabasz_score(states.data, model.labels_)
                ch_scores.append(ch_score)

            if calc_type & 0b0001:
                st_score = check_steady(data, K, slide_window)
                st_scores.append(st_score)

    return sse_scores, sc_scores, ch_scores, st_scores


def monte_carlo(data, states, action_spliter, data_type='env', cluster_type='birch', calc_type=0b1111, K_range=(10, 60),
                slide_window=5, epochs=1, parallel=False):
    K_min, K_max = K_range
    matx_sse = np.mat(np.zeros((epochs, K_max - K_min)))
    matx_sc = np.mat(np.zeros((epochs, K_max - K_min)))
    matx_ch = np.mat(np.zeros((epochs, K_max - K_min)))
    matx_st = np.mat(np.zeros((epochs, K_max - K_min)))
    utils.init_memo(K_max, slide_window, slide_window)

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

                print(f'epoch {i}')
                sse_scores, sc_scores, ch_scores, st_scores = try_K(epochs, i, K_min, K_max, data, states,
                                                                    action_spliter=action_spliter,
                                                                    data_type=data_type,
                                                                    cluster_type=cluster_type,
                                                                    calc_type=calc_type,
                                                                    slide_window=slide_window,
                                                                    parallel=True)
                if calc_type & 0b1000:
                    matx_sse[i, :] = sse_scores
                if calc_type & 0b0100:
                    matx_sc[i, :] = sc_scores
                if calc_type & 0b0010:
                    matx_ch[i, :] = ch_scores
                if calc_type & 0b0001:
                    matx_st[i, :] = st_scores

        # 创建n个线程并运行
        n_threads = 8
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
            sse_scores, sc_scores, ch_scores, st_scores = try_K(epochs, i, K_min, K_max, data, states,
                                                                action_spliter=action_spliter,
                                                                data_type=data_type,
                                                                cluster_type=cluster_type,
                                                                calc_type=calc_type,
                                                                slide_window=slide_window,
                                                                parallel=False)
            if calc_type & 0b1000:
                matx_sse[i, :] = sse_scores
            if calc_type & 0b0100:
                matx_sc[i, :] = sc_scores
            if calc_type & 0b0010:
                matx_ch[i, :] = ch_scores
            if calc_type & 0b0001:
                matx_st[i, :] = st_scores

    fig1 = plt.figure(figsize=(15, 8))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    ax1.set_ylabel('SSE', fontsize=20)
    ax1.set_xlabel('K', fontsize=20)
    ax2.set_ylabel('Value', fontsize=20)
    ax1.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)

    X = range(K_min, K_max)

    fig2 = plt.figure(figsize=(15, 8))
    ax3 = fig2.add_subplot(1, 1, 1)
    ax3.set_xlabel('K', fontsize=20)
    ax3.set_ylabel('Value', fontsize=20)

    mean_sse = None
    mean_sc = None
    mean_ch = None
    mean_st = None
    if calc_type & 0b1000:
        mean_sse = matx_sse.sum(axis=0) / epochs
        np.save('sse.npy', np.asarray(np.vstack((X, mean_sse)).T))
        ax1.plot(X, mean_sse.tolist()[0], marker='o', label='SSE')
    if calc_type & 0b0100:
        mean_sc = matx_sc.sum(axis=0) / epochs
        np.save('silhouette.npy', np.asarray(np.vstack((X, mean_sc)).T))
        ax2.plot(X, mean_sc.tolist()[0], 'r', marker='*', label='Silhouette')
    if calc_type & 0b0010:
        mean_ch = matx_ch.sum(axis=0) / epochs
        mean_ch_norm = mean_ch / max(mean_ch.tolist()[0]) / 3
        np.save('calinski_harabasz.npy', np.asarray(np.vstack((X, mean_ch_norm)).T))
        ax2.plot(X, mean_ch_norm.tolist()[0], 'g', marker='*', label='Calinski Harabasz')
    if calc_type & 0b0001:
        mean_st = matx_st.sum(axis=0) / epochs
        # print(mean_st)
        # print(np.asarray(np.vstack((X, mean_st)).T).shape)
        np.save('steady.npy', np.asarray(np.vstack((X, mean_st)).T))
        ax3.plot(X, mean_st.tolist()[0], 'b', marker='o', label='Steady')

    if calc_type & 0b1000:
        ax1.legend(loc='lower left', fontsize=20)
    if calc_type & 0b0110:
        ax2.legend(loc='upper right', fontsize=20)
    if calc_type & 0b0001:
        ax3.legend(loc='upper right', fontsize=20)

    if calc_type & 0b1110:
        fig1.savefig(f'./{data_type}_sse_sc_ch.png')
    if calc_type & 0b0001:
        fig2.savefig(f'./{data_type}_steady.png')

    return mean_sse, mean_sc, mean_ch, mean_st


def check_raw_data_steady(data, slide_window):
    steady = 0
    total_n = 0
    states = []
    for i in range(len(data)):
        if data[i].state.state_type == 0 or data[i].state.state_type == 1:
            states = []
            continue
        states.append(data[i].state.to_cluster())
        # print(tags)
        if len(states) == slide_window:
            # center = np.mean(states, axis=0)
            # 两两之间的距离
            dists = []
            for j in range(len(states)):
                for k in range(j + 1, len(states)):
                    dists.append(np.linalg.norm(np.array(states[j]) - np.array(states[k])))
            steady += np.mean(dists)
            total_n += 1
            states.pop(0)
    return steady / total_n


def shuffle_raw_data(data):
    """
    shuffle data, 保持state.start和state.end的位置，将中间的数据打乱
    """
    _data = copy.deepcopy(data)
    start_end_pairs = []
    start = 0
    for i in range(len(_data)):
        if _data[i].state.state_type == 1:
            start_end_pairs.append((start, i))
            start = i + 1
    for pair in start_end_pairs:
        np.random.shuffle(_data[pair[0]:pair[1]])
    return _data


def cluster_compare(data, states, action_spliter, data_type='env', K_range=(10, 60), epochs=1, parallel=True):
    # 1. kmeans
    # 2. mini batch kmeans
    # 3. chameleon
    # 4. birch
    # 5. optics
    # 6. clique
    algo_lst = ['kmeans', 'mini_batch_kmeans', 'agglomerative', 'birch']
    algos = [(algo_lst[0], 'r'), (algo_lst[1], 'g'), (algo_lst[2], 'y'), (algo_lst[3], 'b')]
    mean_sts = []
    for algo, color in algos:
        _, _, _, mean_st = monte_carlo(data, states, action_spliter=action_spliter, data_type=data_type, epochs=epochs,
                                       K_range=K_range, calc_type=0b0001, parallel=parallel, cluster_type=algo)
        mean_sts.append(mean_st)

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('K', fontsize=20)
    ax.set_ylabel('Value', fontsize=20)
    ax.tick_params(labelsize=20)

    X = range(K_range[0], K_range[1])
    for i in range(len(algos)):
        ax.plot(X, mean_sts[i].tolist()[0], marker='o', label=algos[i][0], color=algos[i][1])

    ax.legend(loc='lower left', fontsize=20)

    fig.savefig(f'./cluster_compare.png')


def get_action_range(env_data, policy_data):
    """
    获取动作的范围
    policy_data: {state_tag: {action_dict, reward}}
    """
    res = {}
    headers = env_data[0].action.headers
    for header in headers:
        res[header] = [10e5, -10e5]

    for data in [env_data, policy_data]:
        for i in range(len(data)):
            if data[i].state.state_type == 0 or data[i].state.state_type == 1:
                continue
            action_dict = data[i].action.to_dict()
            for header in headers:
                if action_dict[header] < res[header][0]:
                    res[header][0] = action_dict[header]
                if action_dict[header] > res[header][1]:
                    res[header][1] = action_dict[header]

    return res


def parse_code_from_data(K, env_data, policy_data, model, action_spliter, save=False, filename='./code'):
    env_graph = Graph(env_data, K, action_spliter=action_spliter)
    env_graph.gen_nodes()
    env_graph.gen_edges()

    policy_graph = Graph(policy_data, K, action_spliter=action_spliter)
    policy_graph.gen_nodes()
    policy_graph.gen_edges()

    _parser = prism_parser.PrismParser(K, env_graph, policy_graph, props_dict=config['property'])
    retry = 0
    while retry < 10:
        try:
            code = _parser.parse(save=save, file_path=f'{filename}.prism')
            property = _parser.gen_property_file(avg_step=config['data']['policy']['avg_step'], save=save, file_path=f'{filename}.props')
            # print(code)
            break
        except PermissionError:
            retry += 1
            print(f'PermissionError. retry {retry}')
            sleep(2)
            continue
    if retry == 10:
        raise PermissionError


def execute_prism_code(prism_file_path, props_file_path):
    project_dir = os.getcwd()
    os.chdir(prism_path)
    # print(os.getcwd())
    process = subprocess.Popen(['prism.bat', os.path.join('D:/University/project/StateAbstract', prism_file_path),
                                os.path.join('D:/University/project/StateAbstract', props_file_path)],
                               stdout=subprocess.PIPE)

    output, error = process.communicate()
    output = output.decode('utf-8')

    process.terminate()
    os.chdir(project_dir)
    return output, error


def get_info_from_output(output):
    try:
        type_str = re.findall(r'Type:\s+(.*)', output)[0]
        states_str = re.findall(r'States:\s+(\d+)', output)[0]
        transitions_str = re.findall(r'Transitions:\s+(.*)', output)[0]
        choices_str = re.findall(r'Choices:\s+(.*)', output)[0]
        result_str = re.findall(r'Result:\s+(\d+\.\d+)', output)
        # print(result_str)
    except IndexError:
        print(output)
        res = {}
        for prop in config['property']['name']:
            res[prop] = 0
        return res

    type_str = type_str.strip()
    states_val = int(re.sub(r'\D+', '', states_str))
    transitions_val = int(re.sub(r'\D', '', transitions_str))
    choices_val = int(re.sub(r'\D', '', choices_str))

    res = {
        'type': type_str,
        'states': states_val,
        'transitions': transitions_val,
        'choices': choices_val,
    }

    for i, prop_name in enumerate(config['property']['name']):
        property = utils.class_from_path(config['property']['class'][i])(prop_name)
        res[prop_name] = property.get_value(result_str, i)

    return res


def execute_prism_codes(prism_file_paths, parallel=False):
    if parallel:
        record_lock = threading.Lock()
        record = []
        project_dir = os.getcwd()

        def run_task(K, acc, steer):
            # 执行 PRISM 代码
            output, error = execute_prism_code(prism_file_path=f'tmp/code_{K}_{acc}_{steer}.prism',
                                               props_file_path=f'tmp/code_{K}_{acc}_{steer}.props')

            # 从 output 中获取 result
            output_info = get_info_from_output(output)
            with record_lock:
                record.append((K, acc, steer, output_info['episode_reward']))

        def worker(task_queue):
            while True:
                try:
                    K, acc, steer = task_queue.get_nowait()
                except queue.Empty:
                    break
                run_task(K, acc, steer)
                task_queue.task_done()

        def start_workers(task_queue, num_workers=4):
            for i in range(num_workers):
                t = threading.Thread(target=worker, args=(task_queue,))
                t.daemon = True
                t.start()

        def add_tasks(prism_file_paths):
            task_queue = queue.Queue()
            for prism_file_path in os.listdir(prism_file_paths):
                if prism_file_path.endswith('.prism'):
                    # 文件名格式 code_K_acc_steer.prism
                    K, acc, steer = os.path.splitext(prism_file_path)[0].split('_')[1:]
                    task_queue.put((int(K), float(acc), float(steer)))
            return task_queue

        task_queue = add_tasks(prism_file_paths)

        num_workers = 10
        start_workers(task_queue, num_workers=num_workers)

        task_queue.join()
        os.chdir(project_dir)
    else:
        record = []
        for prism_file_path in os.listdir(prism_file_paths):
            if prism_file_path.endswith('.prism'):
                # 文件名格式 code_K_acc_steer.prism
                K, acc, steer = prism_file_path.split('_')[1:]

                # 执行 PRISM 代码
                output, error = execute_prism_code(prism_file_path=prism_file_path,
                                                   props_file_path=f'tmp/code_{K}_{acc}_{steer}.props')
                # 从 output 中获取 result
                output_info = get_info_from_output(output)
                record.append((int(K), float(acc), float(steer), output_info['episode_reward']))

    print(os.getcwd())
    record.sort(key=lambda x: x[0], reverse=True)
    np.save('record.npy', record)
    K, acc, steer, reward = zip(*record)
    print(f'K: {K}')
    print(f'reward: {reward}')

    # if len(granularity_range['acc']) == 1 and len(granularity_range['steer']) == 1:
    #     # 创建 2D 图像
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #
    #     ax.plot(K, reward)
    #     for x, y in zip(K, reward):
    #         ax.text(x, y, f'{y:.2f}')
    #     # plt.axhline(y=episode_reward, linestyle='--')
    #     # ax.text(0, episode_reward, f'{episode_reward:.2f}')
    #     ax.set_xlabel('K')
    #     ax.set_ylabel('reward')
    #     plt.title('Episode Reward')
    #     fig.savefig(f'plot_Episode Reward.png')
    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 使用颜色编码表示第四个变量
    sc = ax.scatter(acc, steer, K, c=reward, cmap='viridis')
    plt.colorbar(sc)

    ax.set_xlabel('acc')
    ax.set_ylabel('steer')
    ax.set_zlabel('K')
    plt.title('Four-dimensional plot')
    fig.savefig(f'plot_Four-dimensional plot.png')

    # plt.show()


def gen_prism_codes(env_data, env_states, policy_data, policy_states, K_range=(5, 40),
                    granularity_range=None, parallel=False):
    if granularity_range is None:
        granularity_range = {'acc': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5], 'steer': [0.01, 0.02, 0.05, 0.1]}

    raw_action_range = get_action_range(env_data, policy_data)

    cluster_type = 'birch'

    if parallel:
        def run_task(env_data, policy_data, K, acc, steer, model):
            print(f'K: {K}, acc: {acc}, steer: {steer}')
            granularity = {'acc': acc, 'steer': steer}
            action_spliter = ActionSpliter(action_ranges=raw_action_range, granularity=granularity)
            set_label(env_data, model, K, cluster_type, 'env')
            set_label(policy_data, model, K, cluster_type, 'policy')
            try:
                parse_code_from_data(K=K, env_data=env_data, policy_data=policy_data,
                                     model=model, action_spliter=action_spliter,
                                     save=True, filename=f'tmp/code_{K}_{acc}_{steer}')
                return True
            except PermissionError:
                return False

        def worker(task_queue, model_cache, env_data, env_states, policy_data, policy_states):
            while True:
                K, acc, steer = task_queue.get()

                if K in model_cache:
                    model = model_cache[K]
                else:
                    model = _cluster(K, env_states, cluster_type)
                    model_cache[K] = model
                done = run_task(env_data, policy_data, K, acc, steer, model)
                if not done:
                    print(f'K: {K}, acc: {acc}, steer: {steer} failed')
                    task_queue.put((K, acc, steer))
                task_queue.task_done()

        def start_workers(task_queue, model_cache, env_data, env_states, policy_data, policy_states, num_workers=4):
            for i in range(num_workers):
                t = threading.Thread(target=worker, args=(task_queue, model_cache,
                                                          copy.deepcopy(env_data),
                                                          copy.deepcopy(env_states),
                                                          copy.deepcopy(policy_data),
                                                          copy.deepcopy(policy_states)))
                t.daemon = True
                t.start()

        def add_tasks(K_range, granularity_range):
            task_queue = queue.Queue()
            for K in range(K_range[0], K_range[1]):
                for acc in granularity_range['acc']:
                    for steer in granularity_range['steer']:
                        task_queue.put((K, acc, steer))
            return task_queue

        task_queue = add_tasks(K_range, granularity_range)
        model_cache = {}

        num_workers = 10
        start_workers(task_queue, model_cache, env_data, env_states, policy_data, policy_states,
                      num_workers=num_workers)

        task_queue.join()
    else:
        # 生成 PRISM 代码
        for K in range(K_range[0], K_range[1]):
            model = _cluster(K, env_states, cluster_type)
            set_label(env_data, model, K, cluster_type, 'env')
            set_label(policy_data, model, K, cluster_type, 'policy')
            for acc in granularity_range['acc']:
                for steer in granularity_range['steer']:
                    granularity = {'acc': acc, 'steer': steer}
                    action_spliter = ActionSpliter(action_ranges=raw_action_range, granularity=granularity)
                    parse_code_from_data(K=K, env_data=env_data, policy_data=policy_data,
                                         model=model, action_spliter=action_spliter,
                                         save=True, filename=f'tmp/code_{K}_{acc}_{steer}')


if __name__ == '__main__':
    start_time = time.time()
    print(f'start_time: {datetime.datetime.now()}')
    # test K
    env_data_raw, env_data, env_states = load_data(config['data'], data_type='env')

    # 验证原始数据有稳定性
    # raw_steady = check_raw_data_steady(env_data, 5)
    # 验证打乱后的数据没有稳定性
    # _data = shuffle_raw_data(env_data)
    # random_steady = check_raw_data_steady(_data, 5)
    # print(f'random_steady: {random_steady}')
    # print(f'raw_steady: {raw_steady}')

    action_spliter = ActionSpliter(action_ranges=get_action_range(env_data, env_data),
                                   granularity={'acc': config['action']['granularity']['acc'],
                                                'steer': config['action']['granularity']['steer']})
    # 对比聚类算法
    # cluster_compare(env_data, env_states, action_spliter=action_spliter, data_type='env', K_range=(10, 20), epochs=5, parallel=True)

    # 求 K 的最佳值
    monte_carlo(env_data, env_states, action_spliter, data_type='env', K_range=(10, 48), calc_type=0b1111,
                slide_window=4, epochs=10, parallel=False)

    # 可视化聚类结果
    # model = _cluster(45, env_states, cluster_type='birch')
    # utils.cluster_visualize(model, env_states['data'], display_type='pca', n_components=2, display_size='normal')

    # 单个运行
    # K = config['cluster']['K']
    # cluster_type = config['cluster']['type']
    # env_data, env_states = load_data(config['data'], data_type='env')
    # policy_data, policy_states = load_data(config['data'], data_type='policy')
    # model = _cluster(K, env_states, cluster_type=cluster_type)
    # set_label(env_data, model, K, cluster_type, 'env')
    # set_label(policy_data, model, K, cluster_type, 'policy')
    # policy_info = utils.get_info_from_data(policy_data, config['property'])
    # print(policy_info)
    # action_spliter = ActionSpliter(action_ranges=get_action_range(env_data, policy_data),
    #                                granularity={'acc': config['action']['granularity']['acc'], 'steer': config['action']['granularity']['steer']})
    # config['data']['policy'].update(policy_info)
    # parse_code_from_data(K=K, env_data=env_data, policy_data=policy_data,
    #                      model=model, action_spliter=action_spliter,
    #                      save=True, filename=f'./code')
    # output, error = execute_prism_code(prism_file_path=f'./code.prism', props_file_path=f'./code.props')
    # output_info = get_info_from_output(output)
    # print(output_info)
    # result_error = utils.compare_result_error(policy_info, output_info, config['property'])
    # print(result_error)

    # env_data_raw, env_data, env_states = load_data(config['data'], data_type='env')
    # policy_data_raw, policy_data, policy_states = load_data(config['data'], data_type='policy')
    # policy_info = utils.get_info_from_data(policy_data_raw, config['property'])
    # config['data']['policy'].update(policy_info)
    # print(config['data']['policy'])
    # gen_prism_codes(env_data, env_states, policy_data, policy_states, K_range=(10, 48),
    #                 granularity_range={'acc': [0.01], 'steer': [0.01]}, parallel=True)
    # execute_prism_codes(prism_file_paths='./tmp', parallel=True)

    # draw_graph(env_graph, K)
    # draw_heatmap(policy_data)
    # _parser = uppaal_parser.UppaalParser(graph)
    # xml_tree = _parser.to_xml()
    # uppaal_parser.write(xml_tree, 'uppaal')
    end_time = time.time()
    print(f'end_time: {datetime.datetime.now()}')
    print(f'cost time: {end_time - start_time}s')
    pass
