import json
import numpy as np
import matplotlib.pyplot as plt

data_file = '20000_random_movement_data_2023_11_26_21_51.json'
# data_file = 'MPC_20000_movement_data_2023_11_27_00_42.json'
actions = 'action'
curr_position_data = 'current position'
curr_orientation_data = 'current orientation'
n_1_data = 'n_1_state'
n_2_data = 'n_2_state'
n_3_data = 'n_3_state'
n_4_data = 'n_4_state'
n_5_data = 'n_5_state'

data_file = open(data_file)
data_load = json.load(data_file)
# Individual Category Arrays

body_length = 0.2884
body_width = 0.2216

mpc_cost = False

if mpc_cost:
    action_data = data_load[actions]
    position_data = np.asarray(data_load[curr_position_data])
    orientation_data = data_load[curr_orientation_data]
    cost = 'cost'
    cost_data = np.asarray(data_load[cost])
    steps = np.arange(len(cost_data))

    ######################################################################################################

    plt.xlim(0, np.max(steps))
    plt.xlabel('Steps')
    plt.ylim(np.min(cost_data), np.max(cost_data))
    plt.ylabel('Cost')
    plt.title("Cost Progression Traj_1")
    plt.plot(steps, cost_data, 'g-', label='Cost')
    plt.legend()
    plt.show()

    ######################################################################################################

    # plt.xlim(np.min(position_data[:, 0, :], axis=0), np.max(position_data[:, 0, :], axis=0))
    plt.xlim(-0.15, 0.15)
    plt.xlabel('X [m]')
    # plt.ylim(np.min(position_data[:, 1, :], axis=0), np.max(position_data[:, 1, :], axis=0))
    plt.ylim(-0.15, 0.15)
    plt.ylabel('Y [m]')
    plt.title('MPC Path ' f'{max(steps)+1}' ' Steps')
    plt.plot(position_data[:, 0, :], position_data[:, 1, :], 'b-', label='Trajectory')
    plt.legend()
    plt.show()

else:

    position_data = np.asarray(data_load[curr_position_data])
    orientation_data = data_load[curr_orientation_data]
    n_1 = data_load[n_1_data]
    n_5 = np.asarray(data_load[n_5_data])
    print(n_5.shape)
    n_5_position = n_5[:, 0:3]
    # array = np.concatenate((position_data, orientation_data, n_1, n_5), axis=1)
    # input = np.concatenate((position_data, orientation_data), axis=1)

    ######################################################################################################

    label_list = ['X Position', 'Y Position', 'Z Position', 'X Orientation', 'Y Orientation', 'Z Orientation',
                  'W Orientation']

    # PLOT FULL TRAJECTORY
    plt.xlim(np.min(position_data[0:200, 0], axis=0), np.max(position_data[0:200, 0], axis=0))
    plt.xlabel(f'{label_list[0]}')
    plt.ylim(np.min(position_data[0:200, 1], axis=0), np.max(position_data[0:200, 1], axis=0))
    plt.ylabel(f'{label_list[1]}')
    plt.title("Trajectory")
    plt.plot(position_data[0:200, 0], position_data[0:200, 1], '-', 'k', label='Ground Truth')
    plt.plot(n_5_position[0:200, 0], n_5_position[0:200, 1], '-.', 'b', label='N-5')
    plt.legend()
    plt.show()

    # PLOT N-X TRAJECTORY
    # plt.xlim(np.min(n_5_position[0:, 0], axis=0), np.max(n_5_position[0:, 0], axis=0))
    # plt.xlabel(f'{label_list[0]}')
    # plt.ylim(np.min(n_5_position[0:, 1], axis=0), np.max(n_5_position[0:, 1], axis=0))
    # plt.ylabel(f'{label_list[1]}')
    # plt.title("N-5 Trajectory")
    # plt.plot(n_5_position[0:200, 0], n_5_position[0:200, 1], 'b-', label='Ground Truth')
    # plt.legend()
    # plt.show()

###################################################################################################