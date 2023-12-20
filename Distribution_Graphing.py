import json
import numpy as np
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()

data_file_1 = 'MPC_20000_movement_data_2023_11_27_00_42.json'

data_1 = open(data_file_1)
data_load_1 = json.load(data_1)

mean = 'mean'
var = 'variance'

mean_data = np.reshape(np.asarray(data_load_1[mean]), (20000, 12))
var_data = np.reshape(np.asarray(data_load_1[var]), (20000, 12))
steps = np.arange(mean_data.shape[0])

lower = mean_data-var_data
upper = mean_data-var_data

lower_flat = np.ndarray.flatten(lower)
upper_flat = np.ndarray.flatten(upper)

lim = 6

plt.xlim(0, lim)
plt.xlabel('Steps')
plt.ylim(-0.2, 0.2)

plt.ylabel('Variance in Position Command [rad]')
plt.title("StDev of Motor Commands at each Step")

# plt.ylabel('Mean Position Command [rad]')
# plt.title("Mean Motor Commands at each Step")

plt.plot(np.arange(lim), var_data[0:lim, 0], color='g', label='Motor 1')
plt.plot(np.arange(lim), var_data[0:lim, 3], color='m', label='Motor 4')
plt.plot(np.arange(lim), var_data[0:lim, 8], color='r', label='Motor 9')
plt.plot(np.arange(lim), var_data[0:lim, 10], color='b', label='Motor 11')
plt.plot(np.arange(lim), var_data[0:lim, 11], color='k', label='Motor 12')

# plt.plot(steps, lower[:, 0], label='Upper Bounds')
# plt.plot(steps, upper[:, 0], label='Lower Bounds')

plt.legend()
plt.show()





