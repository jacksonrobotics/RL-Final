import numpy as np
import json
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = 4,3
from matplotlib.animation import FuncAnimation

data_file = 'MPC_1000_movement_data_2023_11_15_09_09.json'
# x_position = 'X predictions_1'
# y_position = 'Y predictions_1'
# actions = 'action'
curr_position_data = 'current position'
curr_orientation_data = 'current orientation'

data_file = open(data_file)
data_load = json.load(data_file)
# Individual Category Arrays
position_data = data_load[curr_position_data]
orientation_data = data_load[curr_orientation_data]
array = np.concatenate((position_data, orientation_data), axis=1)
x = array[:, 0].tolist()
y = array[:, 1].tolist()
# x = data_load[x_position]
# y = data_load[y_position]
# print(x[1])

def path(idx):
    return x[idx], y[idx]

fig = plt.figure()

ax = plt.axes(xlim=(-0.05, 0.21), ylim=(-.2, 0.27))
# set equal aspect such that the circle is not shown as ellipse
# create a point in the axes
point, = ax.plot(0, 1, 'r', marker="o")
plt.plot(x, y, 'b-')
# Updating function, to be repeatedly called by the animation
def update(i):
    # obtain point coordinates
    x, y = path(i)
    # set point's coordinates
    point.set_data([x], [y])
    return point,

# create animation with 10ms interval, which is repeated,
ani = FuncAnimation(fig, update, interval=60, blit=True, repeat=True, frames=10000)
fig.suptitle('MPC Traj_3', fontsize=14)
plt.show()