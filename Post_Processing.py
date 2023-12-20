import json
import numpy as np
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()

# Reference: https://stackoverflow.com/questions/12957582/plot-yerr-xerr-as-shaded-region-rather-than-error-bars

# Colors: https://www.color-hex.com/ 

data_file = "Nov27_15-06-11_pse-bc298-dt11.egr.duke.edu.json"

data = open(data_file)
data_load = json.load(data)

steps = np.reshape(np.asarray(data_load)[:, 1], (40, 1))
loss = np.reshape(np.asarray(data_load)[:, 2], (40, 1))

# Mean Normal Validation Loss
# data_file_0_1 = "N Recursive/Nov26_21-39-11_pse-bc298-dt11.egr.duke.edu(1).json"
data_file_0_2 = "N Recursive/Nov26_21-42-16_pse-bc298-dt11.egr.duke.edu(1).json"
# data_file_0_3 = "N Recursive/Nov26_21-45-17_pse-bc298-dt11.egr.duke.edu(1).json"
#
# data_file_1_1 = "N-1 Recursive/Nov26_21-28-19_pse-bc298-dt11.egr.duke.edu(1).json"
# data_file_1_2 = "N-1 Recursive/Nov26_21-31-13_pse-bc298-dt11.egr.duke.edu(1).json"
# data_file_1_3 = "N-1 Recursive/Nov26_21-34-02_pse-bc298-dt11.egr.duke.edu(1).json"
#
# data_file_2_1 = "N-2 Recursive/Nov26_21-16-35_pse-bc298-dt11.egr.duke.edu(1).json"
# data_file_2_2 = "N-2 Recursive/Nov26_21-19-01_pse-bc298-dt11.egr.duke.edu(1).json"
# data_file_2_3 = "N-2 Recursive/Nov26_21-21-54_pse-bc298-dt11.egr.duke.edu(1).json"
#
# data_file_3_1 = "N-3 Recursive/Nov26_20-35-43_pse-bc298-dt11.egr.duke.edu(1).json"
# data_file_3_2 = "N-3 Recursive/Nov26_20-39-37_pse-bc298-dt11.egr.duke.edu(1).json"
# data_file_3_3 = "N-3 Recursive/Nov26_20-42-28_pse-bc298-dt11.egr.duke.edu(1).json"
#
# data_file_4_1 = "N-4 Recursive/Nov26_20-21-27_pse-bc298-dt11.egr.duke.edu(1).json"
data_file_4_2 = "N-4 Recursive/Nov26_20-25-10_pse-bc298-dt11.egr.duke.edu(1).json"
# data_file_4_3 = "N-4 Recursive/Nov26_20-28-40_pse-bc298-dt11.egr.duke.edu(1).json"
#
# data_file_5_1 = "N-5 Recursive/Nov26_20-08-41_pse-bc298-dt11.egr.duke.edu(1).json"
data_file_5_2 = "N-5 Recursive/Nov26_20-12-03_pse-bc298-dt11.egr.duke.edu(1).json"
# data_file_5_3 = "N-5 Recursive/Nov26_20-15-23_pse-bc298-dt11.egr.duke.edu(1).json"

#########################################################
# data_0_1 = open(data_file_0_1)
# data_load_0_1 = json.load(data_0_1)
#
data_0_2 = open(data_file_0_2)
data_load_0_2 = json.load(data_0_2)
#
# data_0_3 = open(data_file_0_3)
# data_load_0_3 = json.load(data_0_3)
#
# loss_0_1 = np.reshape(np.asarray(data_load_0_1)[:, 2], (40, 1))
loss_0_2 = np.reshape(np.asarray(data_load_0_2)[:, 2], (40, 1))
# loss_0_3 = np.reshape(np.asarray(data_load_0_3)[:, 2], (40, 1))
#
# loss_0 = np.concatenate((loss_0_1, loss_0_2, loss_0_3), axis=1)
#
# mean_loss_0 = np.reshape(np.mean(loss_0, axis=1), (40, 1))
# var_loss_0 = np.reshape(np.std(loss_0, axis=1), (40, 1))
#
# lower_0 = mean_loss_0-var_loss_0
# upper_0 = mean_loss_0+var_loss_0
# #
# lower_flat_0 = np.ndarray.flatten(lower_0)
# upper_flat_0 = np.ndarray.flatten(upper_0)
#
#
# # #########################################################
# data_1_1 = open(data_file_1_1)
# data_load_1_1 = json.load(data_1_1)
#
# data_1_2 = open(data_file_1_2)
# data_load_1_2 = json.load(data_1_2)
#
# data_1_3 = open(data_file_1_3)
# data_load_1_3 = json.load(data_1_3)
#
# steps = np.reshape(np.asarray(data_load_1_1)[:, 1], (40, 1))
# loss_1_1 = np.reshape(np.asarray(data_load_1_1)[:, 2], (40, 1))
# loss_1_2 = np.reshape(np.asarray(data_load_1_2)[:, 2], (40, 1))
# loss_1_3 = np.reshape(np.asarray(data_load_1_3)[:, 2], (40, 1))
#
# loss_1 = np.concatenate((loss_1_1, loss_1_2, loss_1_3), axis=1)
#
# mean_loss_1 = np.reshape(np.mean(loss_1, axis=1), (40, 1))
# var_loss_1 = np.reshape(np.std(loss_1, axis=1), (40, 1))
#
# lower_1 = mean_loss_1-var_loss_1
# upper_1 = mean_loss_1+var_loss_1
# #
# lower_flat_1 = np.ndarray.flatten(lower_1)
# upper_flat_1 = np.ndarray.flatten(upper_1)
#
# #########################################################
# data_2_1 = open(data_file_2_1)
# data_load_2_1 = json.load(data_2_1)
#
# data_2_2 = open(data_file_2_2)
# data_load_2_2 = json.load(data_2_2)
#
# data_2_3 = open(data_file_2_3)
# data_load_2_3 = json.load(data_2_3)
#
# # steps = np.reshape(np.asarray(data_load_2_1)[:, 1], (40, 1))
# loss_2_1 = np.reshape(np.asarray(data_load_2_1)[:, 2], (40, 1))
# loss_2_2 = np.reshape(np.asarray(data_load_2_2)[:, 2], (40, 1))
# loss_2_3 = np.reshape(np.asarray(data_load_2_3)[:, 2], (40, 1))
#
# loss_2 = np.concatenate((loss_2_1, loss_2_2, loss_2_3), axis=1)
#
# mean_loss_2 = np.reshape(np.mean(loss_2, axis=1), (40, 1))
# var_loss_2 = np.reshape(np.std(loss_2, axis=1), (40, 1))
#
# lower_2 = mean_loss_2-var_loss_2
# upper_2 = mean_loss_2+var_loss_2
# #
# lower_flat_2 = np.ndarray.flatten(lower_2)
# upper_flat_2 = np.ndarray.flatten(upper_2)
# #
# # #########################################################
# data_3_1 = open(data_file_3_1)
# data_load_3_1 = json.load(data_3_1)
#
# data_3_2 = open(data_file_3_2)
# data_load_3_2 = json.load(data_3_2)
#
# data_3_3 = open(data_file_3_3)
# data_load_3_3 = json.load(data_3_3)
#
# loss_3_1 = np.reshape(np.asarray(data_load_3_1)[:, 2], (40, 1))
# loss_3_2 = np.reshape(np.asarray(data_load_3_2)[:, 2], (40, 1))
# loss_3_3 = np.reshape(np.asarray(data_load_3_3)[:, 2], (40, 1))
#
# loss_3 = np.concatenate((loss_3_1, loss_3_2, loss_3_3), axis=1)
#
# mean_loss_3 = np.reshape(np.mean(loss_3, axis=1), (40, 1))
# var_loss_3 = np.reshape(np.std(loss_3, axis=1), (40, 1))
#
# lower_3 = mean_loss_3-var_loss_3
# upper_3 = mean_loss_3+var_loss_3
# #
# lower_flat_3 = np.ndarray.flatten(lower_3)
# upper_flat_3 = np.ndarray.flatten(upper_3)
#
# #########################################################
# data_4_1 = open(data_file_4_1)
# data_load_4_1 = json.load(data_4_1)
#
data_4_2 = open(data_file_4_2)
data_load_4_2 = json.load(data_4_2)
#
# data_4_3 = open(data_file_4_3)
# data_load_4_3 = json.load(data_4_3)
#
# # steps = np.reshape(np.asarray(data_load_4_1)[:, 1], (40, 1))
# loss_4_1 = np.reshape(np.asarray(data_load_4_1)[:, 2], (40, 1))
loss_4_2 = np.reshape(np.asarray(data_load_4_2)[:, 2], (40, 1))
# loss_4_3 = np.reshape(np.asarray(data_load_4_3)[:, 2], (40, 1))
#
# loss_4 = np.concatenate((loss_4_1, loss_4_2, loss_4_3), axis=1)
#
# mean_loss_4 = np.reshape(np.mean(loss_4, axis=1), (40, 1))
# var_loss_4 = np.reshape(np.std(loss_4, axis=1), (40, 1))
#
# lower_4 = mean_loss_4-var_loss_4
# upper_4 = mean_loss_4+var_loss_4
# #
# lower_flat_4 = np.ndarray.flatten(lower_4)
# upper_flat_4 = np.ndarray.flatten(upper_4)
#
# # #########################################################
# data_5_1 = open(data_file_5_1)
# data_load_5_1 = json.load(data_5_1)
#
data_5_2 = open(data_file_5_2)
data_load_5_2 = json.load(data_5_2)
#
# data_5_3 = open(data_file_5_3)
# data_load_5_3 = json.load(data_5_3)
#
# # steps = np.reshape(np.asarray(data_load_5_1)[:, 1], (40, 1))
# loss_5_1 = np.reshape(np.asarray(data_load_5_1)[:, 2], (40, 1))
loss_5_2 = np.reshape(np.asarray(data_load_5_2)[:, 2], (40, 1))
# loss_5_3 = np.reshape(np.asarray(data_load_5_3)[:, 2], (40, 1))
#
# loss_5 = np.concatenate((loss_5_1, loss_5_2, loss_5_3), axis=1)
#
# mean_loss_5 = np.reshape(np.mean(loss_5, axis=1), (40, 1))
# var_loss_5 = np.reshape(np.std(loss_5, axis=1), (40, 1))
#
# lower_5 = mean_loss_5-var_loss_5
# upper_5 = mean_loss_5+var_loss_5
# #
# lower_flat_5 = np.ndarray.flatten(lower_5)
# upper_flat_5 = np.ndarray.flatten(upper_5)
#########################################################

plt.xlim(0, np.max(steps))
plt.xlabel('Epochs [-]')
plt.ylim(0, np.max(loss_0_2))
plt.ylabel('MSE Loss [m]')
# plt.title("Mean Validation Loss (N-2)")
plt.title("Recursive Loss Comparisons")
plt.plot(steps, loss[:, 0], color='k', label='N-5, 200K Model')
plt.plot(steps, loss_0_2[:, 0], color='g', label='N')
plt.plot(steps, loss_4_2[:, 0], color='b', label='N-4')
plt.plot(steps, loss_5_2[:, 0], color='r', label='N-5')

# plt.plot(steps, mean_loss_0[:, 0], color='g', label='Mean (N)')
# plt.plot(steps, lower_0[:, 0], '--', color='g')
# plt.plot(steps, upper_0[:, 0], '--', color='g')
# plt.fill_between(np.ndarray.flatten(steps), lower_flat_0, upper_flat_0, alpha=0.5, edgecolor='#22b93b', facecolor='#affabb')

# plt.plot(steps, mean_loss_1[:, 0], color='m', label='Mean (N-1)')
# plt.plot(steps, lower_1[:, 0], '--', color='m')
# plt.plot(steps, upper_1[:, 0], '--', color='m')
# plt.fill_between(np.ndarray.flatten(steps), lower_flat_1, upper_flat_1, alpha=0.5, edgecolor='#c62ea7', facecolor='#ffbaf1')
#
# plt.plot(steps, mean_loss_2[:, 0], color='y', label='Mean (N-2)')
# plt.plot(steps, lower_2[:, 0], '--', color='y')
# plt.plot(steps, upper_2[:, 0], '--', color='y')
# plt.fill_between(np.ndarray.flatten(steps), lower_flat_2, upper_flat_2, alpha=0.5, edgecolor='#feeb00', facecolor='#fff8a1')
#
# plt.plot(steps, mean_loss_3[:, 0], color='r', label='Mean (N-3)')
# plt.plot(steps, lower_3[:, 0], '--', color='r')
# plt.plot(steps, upper_3[:, 0], '--', color='r')
# plt.fill_between(np.ndarray.flatten(steps), lower_flat_3, upper_flat_3, alpha=0.5, edgecolor='#eb424e', facecolor='#faafb4')
#
# plt.plot(steps, mean_loss_4[:, 0], color='k', label='Mean (N-4)')
# plt.plot(steps, lower_4[:, 0], '--', color='k')
# plt.plot(steps, upper_4[:, 0], '--', color='k')
# plt.fill_between(np.ndarray.flatten(steps), lower_flat_4, upper_flat_4, alpha=0.5, edgecolor='#555454', facecolor='#b1afaf')
#
# plt.plot(steps, mean_loss_5[:, 0], color='b', label='Mean (N-5)')
# plt.plot(steps, lower_5[:, 0], '--', color='b')
# plt.plot(steps, upper_5[:, 0], '--', color='b')
# plt.fill_between(np.ndarray.flatten(steps), lower_flat_5, upper_flat_5, alpha=0.5, edgecolor='#1a91fa', facecolor='#a0d2fe')

plt.legend()
plt.show()




