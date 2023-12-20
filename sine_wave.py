import math
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma

time = 0
time_step = 1 / 240
joint_num_size = 12
n = 500
scale = 3
a = 0.1
b = 0.5
c = 0.3
gait_array = np.zeros([12])
time_array = np.zeros([1])

# print(np.random.uniform(-1.4, 1.4, size=joint_num_size))
omega = 5



def random_para():
    para = np.zeros(16)
    for i in range(16):
        para[i] = np.random.uniform(-1, 1)
    for i in range(2, 6):
        para[i] *= 2 * np.pi

    return para


def sin_move(time, para, sep=16):
    # print(para)
    s_action = np.zeros(12)
    # print(ti)
    s_action[0] = para[0] * np.sin(time / sep * 2 * np.pi + para[2]) + para[10]  # left   hind
    s_action[3] = para[1] * np.sin(time / sep * 2 * np.pi + para[3]) + para[11]  # left   front
    s_action[6] = para[1] * np.sin(time / sep * 2 * np.pi + para[4]) - para[11]  # right  front
    s_action[9] = para[0] * np.sin(time / sep * 2 * np.pi + para[5]) - para[10]  # right  hind

    s_action[1] = para[6] * np.sin(time / sep * 2 * np.pi + para[2]) - para[12]  # left   hind
    s_action[4] = para[7] * np.sin(time / sep * 2 * np.pi + para[3]) - para[13]  # left   front
    s_action[7] = para[7] * np.sin(time / sep * 2 * np.pi + para[4]) - para[13]  # right  front
    s_action[10] = para[6] * np.sin(time / sep * 2 * np.pi + para[5]) - para[12]  # right  hind

    s_action[2] = para[8] * np.sin(time / sep * 2 * np.pi + para[2]) + para[14]  # left   hind
    s_action[5] = para[9] * np.sin(time / sep * 2 * np.pi + para[3]) + para[15]  # left   front
    s_action[8] = para[9] * np.sin(time / sep * 2 * np.pi + para[4]) + para[15]  # right  front
    s_action[11] = para[8] * np.sin(time / sep * 2 * np.pi + para[5]) + para[14]  # right  hind


    return s_action

for i in range(n):
    param = random_para()
    random_sine_gait = sin_move(time, param)

    # random_a = np.random.uniform(-1.05, 1.05, size=joint_num_size)
    # random_c = np.random.uniform(-1, 1, size=joint_num_size)
    # random_sine_gait = random_a * np.sin(omega * time + random_c)


    gait_array = np.vstack((gait_array, random_sine_gait))
    # print(gait_array.shape)

    time += time_step
    time_array = np.append(time_array, time)

if __name__ == "__main__":
    # gait_array = np.transpose(gait_array)
    # print(gait_array.shape)
    # print(time_array.shape)
    plt.xlim(0, 100)
    plt.xlabel('Steps [-]')
    plt.ylim(-1.55, 1.55)
    plt.ylabel('Joint Angle Commands [rad]')
    plt.plot(np.arange(501), gait_array[:, 0:1], 'b-')
    plt.legend(["Motor 1", "Motor 2", "Motor 3"], loc="upper right")
    plt.title("Self-Modeling Random Motor Commands")

    plt.show()
