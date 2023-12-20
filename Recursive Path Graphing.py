import json
import numpy as np
import matplotlib.pyplot as plt

data_file = '20000_random_movement_data_2023_11_26_21_51.json'

curr_position_data = 'current position'
n_1_data = 'n_1_state'
n_2_data = 'n_2_state'
n_3_data = 'n_3_state'
n_4_data = 'n_4_state'
n_5_data = 'n_5_state'

data_file = open(data_file)
data_load = json.load(data_file)


position_data = np.asarray(data_load[curr_position_data])
n_2 = np.asarray(data_load[n_2_data])
n_5 = np.asarray(data_load[n_5_data])
n_2_position = n_2[:, 0:3]
n_5_position = n_5[:, 0:3]

label_list = ['X Position [m]', 'Y Position [m]', 'Z Position', 'X Orientation', 'Y Orientation', 'Z Orientation', 'W Orientation']

# PLOT FULL TRAJECTORY
plt.xlim(np.min(position_data[0:200, 0], axis=0), np.max(position_data[0:200, 0], axis=0))
plt.xlabel(f'{label_list[0]}')
plt.ylim(np.min(position_data[0:200, 1], axis=0), np.max(position_data[0:200, 1], axis=0))
plt.ylabel(f'{label_list[1]}')
plt.title("Trajectory 1")
plt.plot(position_data[0:200, 0], position_data[0:200, 1], 'k-', label='Ground Truth')
plt.plot(n_2_position[0:200, 0], n_2_position[0:200, 1], 'r-', label='N-2')
plt.plot(n_5_position[0:200, 0], n_5_position[0:200, 1], 'b-', label='N-5')
plt.plot(0, 0, 'go')
plt.legend()
plt.show()

