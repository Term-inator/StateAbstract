import os

import pandas as pd
import matplotlib.pyplot as plt

dir = '../RL-Carla/output_logger/policy-lane-dest150m-after0225-reward300'
filename = 'run-Lane-tag-Reward_epoch_reward.csv'
data = pd.read_csv(os.path.join(dir, filename))

x = data['Step']
y = data['Value']

plt.plot(x, y)
plt.xlabel('Step')
plt.ylabel('Value')
plt.title(filename)
plt.show()
