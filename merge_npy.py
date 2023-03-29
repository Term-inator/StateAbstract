import os

import numpy as np


def merge_prism_experiment(file_list):
    tuples = []
    for file in file_list:
        data = np.load(file)
        t = np.array([v for v in zip(*data)])

        for i in range(len(t[0])):
            tuples.append(tuple(t[:, i]))

    tuples.sort(key=lambda x: x[0])

    # 定义一个字典，用于记录重复的 tuple 和它们的出现次数
    tuple_count = {}

    # 遍历 tuple 列表，统计每个 tuple 的出现次数
    for t in tuples:
        index = t[0]
        if index in tuple_count:
            tuple_count[index] += 1
        else:
            tuple_count[index] = 1

    # 找出重复的 tuple，并输出它们的值
    duplicates = [index for index in tuple_count if tuple_count[index] > 1]
    for index in duplicates:
        print(f"{index} 出现了 {tuple_count[index]} 次")

    # 让用户选择保留哪个数据
    for index in duplicates:
        print(f"请选择保留哪组数据：")
        for i, x in enumerate([i for i in range(len(tuples)) if tuples[i][0] == index]):
            print(f"{i + 1}. {tuples[x]}")
        choice = input()
        while not choice.isdigit() or int(choice) < 1 or int(choice) > len([i for i in range(len(tuples)) if tuples[i][0] == index]):
            print("输入无效，请重新输入")
            choice = input()
        choice_index = [i for i in range(len(tuples)) if tuples[i][0] == index][int(choice) - 1]
        print(choice_index, index)
        print(f"您选择了 {tuples[choice_index]}")
        t = tuples[choice_index]
        tuples.remove(t)
        tuples.append(t)

    tuples.sort(key=lambda x: x[0])
    return tuples


directory = 'output/race_track'

npy_files = ['6-10.npy', '10-30.npy', '31-50.npy', '51-90.npy', '91-100.npy']
files = []

for npy_file in npy_files:
    files.append(os.path.join(directory, npy_file))


data = merge_prism_experiment(files)
np.save(os.path.join(directory, 'K,acc,steer,reward.npy'), data)
K, acc, steer, reward = zip(*data)
print(K)
print(acc)
print(steer)
print(reward)
