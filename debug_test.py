import sys
print("SYS Stuff \n", sys.executable)


# import json
# import torch
# import torch.nn as nn
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, Dataset
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()


# class QuickNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(13, 128),
#             nn.LeakyReLU(),
#             nn.Linear(128, 128),
#             nn.LeakyReLU(),
#             nn.Linear(128, 128),
#             nn.LeakyReLU(),
#             nn.Linear(128, 1),
#         )

#     def forward(self, x):
#         y_predict = self.linear_relu_stack(x)
#         return y_predict


# class MovementDataset(Dataset):
#     def __init__(self, data_file='1500_random_movement_data_2023_10_13_10_57.json',
#                  actions_data='action',
#                  position_data='current position',
#                  orientation_data='current orientation'):
#         self._data_file = open(data_file)
#         self._data_load = json.load(self._data_file)
#         # Individual Category Arrays
#         self._actions_data = self._data_load[actions_data]  # (SAMPLE x 12)
#         self._position_data = self._data_load[position_data]
#         self._orientation_data = self._data_load[orientation_data]
#         # Define Input and Output
#         self._Input = self.getInput(self._actions_data, self._position_data, self._orientation_data)
#         self._Output = self.getOutput(self._position_data, self._orientation_data)
#         self.n_samples = len(self._Input)  # Only returns the number of
#         # SAMPLES for Actions

#     def __getitem__(self, index):
#         X = self._Input[index]
#         y = self._Output[index]
#         return X, y

#     def __len__(self):
#         # len(dataset)
#         return self.n_samples

#     def getInput(self, dataset_1, dataset_2, dataset_3):

#         array = np.concatenate((dataset_1, dataset_2),
#                                axis=1)  # (action_array, position_array, orientation_array)

#         N = array.shape[0]
#         input_array = array[0:-1, 0:13]
#         print("INPUT ARRAY SHAPE: ", input_array.shape)  # (10 x 13), (SAMPLESx19)
#         pre_tensor = torch.from_numpy(input_array.astype(np.float32))
#         input_tensor = pre_tensor.requires_grad_()
#         # print("INPUT ARRAY: ", input_array)
#         return input_tensor  # correct size (19x9), requires_grad = true

#     def getOutput(self, dataset_1, dataset_2):
#         array = np.reshape(dataset_1, (-1, 1))  # (position_array, orientation_array)

#         output_array = array[1:1501, 0]
#         print("OUTPUT ARRAY SHAPE: ", output_array.shape)  #(SAMPLESx1), [XP, YP, ZP, XO, YO, ZO, WO]
#         print("OUTPUT ARRAY: ", output_array)
#         pre_tensor = torch.from_numpy(output_array.astype(np.float32))
#         output_tensor = pre_tensor.requires_grad_()
#         # print("OUTPUT ARRAY", output_array)
#         return output_tensor  # correct size (7x9), requires_grad = true


# # Initialize dataset Class
# crawler1_dataset = MovementDataset()
# # Training and test datasets
# train_set, test_set = torch.utils.data.random_split(crawler1_dataset, [int(0.8 * len(crawler1_dataset)),
#                                                                        len(crawler1_dataset) - int(
#                                                                            0.8 * len(crawler1_dataset))])

# batch_size = 4  # HYPERPARAMETER
# train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# # Initialize the model
# model = QuickNN()

# # Loss and Optimizer
# learning_rate = 0.01  # HYPERPARAMETER
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# # training
# num_epochs = 10  # HYPERPARAMETER
# total_samples = len(train_set)
# n_iterations = math.ceil(total_samples / batch_size)
# print(total_samples, n_iterations)
# previous_prediction = []
# norm_path = np.empty((1, 1))
# recursive_path = np.empty((1, 1))
# ground_truth = np.empty((1, 1))

# for epoch in range(num_epochs):
#     total_train_loss = 0
#     total_norm_val_loss = 0
#     total_recursive_val_loss = 0
#     loss_XP = 0
#     loss_YP = 0
#     loss_ZP = 0
#     loss_XO = 0
#     loss_YO = 0
#     loss_ZO = 0
#     loss_WO = 0
#     mean_loss_XP = []
#     mean_loss_YP = []
#     mean_loss_ZP = []
#     mean_loss_XO = []
#     mean_loss_YO = []
#     mean_loss_ZO = []
#     mean_loss_WO = []

#     for i, (state, y_action) in enumerate(train_dataloader):

#         # zero gradients
#         optimizer.zero_grad()

#         y_predicted = model(state)
#         loss = criterion(y_predicted, y_action)
#         total_train_loss += loss

#         # Compute Loss
#         loss.backward(retain_graph=True)
#         # Update Weights
#         optimizer.step()

#         if (i + 1) % 10 == 0:
#             print(f'iteration: {i + 1}, training loss = {loss.item():.4f}')
#             print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_iterations}')
#     # send to tensorboard
#     writer.add_scalar("Loss/train", total_train_loss / len(train_set), epoch)

#     with torch.no_grad():
#         for i, (state, y_val_action) in enumerate(test_dataloader):
#             # Original Prediction pipeline
#             norm_y_predicted = model(state)
#             norm_y_predicted_array = norm_y_predicted.numpy()
#             # print("CURRENT STATE:   ", state[:, 12:])
#             # print("NEXT STEP:   ", y_val_action)
#             # print("PREDICTED NEXT STEP:    ", norm_y_predicted)
#             norm_path = np.append(norm_path, norm_y_predicted_array, axis=0)
#             ground_truth = np.append(ground_truth, state[:, 12:], axis=0)
#             norm_val_loss = criterion(norm_y_predicted, y_val_action)
#             total_norm_val_loss += norm_val_loss

#             ###################################
#             # The State input and the NEXT STEP are the SAME
#             ###################################

#             # Recursive Prediction Pipeline
#             if i == 0:
#                 recursive_y_predicted = model(state)
#                 recursive_path = np.append(recursive_path, recursive_y_predicted, axis=0)
#                 recursive_val_loss = criterion(recursive_y_predicted, y_val_action)
#                 total_recursive_val_loss += recursive_val_loss
#                 previous_prediction = recursive_y_predicted
#             else:
#                 state[:, 12:] = previous_prediction
#                 recursive_y_predicted = model(state)
#                 recursive_path = np.append(recursive_path, recursive_y_predicted, axis=0)
#                 recursive_val_loss = criterion(recursive_y_predicted, y_val_action)
#                 total_recursive_val_loss += recursive_val_loss
#                 previous_prediction = recursive_y_predicted

#             #############################################################

#             for e in range(norm_y_predicted.shape[0]):
#                 loss_0 = criterion(recursive_y_predicted[e], y_val_action[e])

#                 # Summing the loss for each prediction in batch
#                 loss_XP += loss_0
#                 # loss_YP += loss_1
#                 # loss_ZP += loss_2
#                 # loss_XO += loss_3
#                 # loss_YO += loss_4
#                 # loss_ZO += loss_5
#                 # loss_WO += loss_6
#             # Averaging the loss over num_samples
#             loss_XP_avg = loss_XP / norm_y_predicted.shape[0]
#             # loss_YP_avg = loss_YP / norm_y_predicted.shape[0]
#             # loss_ZP_avg = loss_ZP / norm_y_predicted.shape[0]
#             # loss_XO_avg = loss_XO / norm_y_predicted.shape[0]
#             # loss_YO_avg = loss_YO / norm_y_predicted.shape[0]
#             # loss_ZO_avg = loss_ZO / norm_y_predicted.shape[0]
#             # loss_WO_avg = loss_WO / norm_y_predicted.shape[0]
#             #############################################################

#             # send to tensorboard
#             if (i + 1) % 10 == 0:
#                 print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}, Validation loss = {norm_val_loss.item():.4f}')
#                 # print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}, Val loss X POS = {loss_XP_avg:.4f}')
#                 # print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}, Val loss Y POS = {loss_YP_avg:.4f}')
#                 # print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}, Val loss Z POS = {loss_ZP_avg:.4f}')
#                 # print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}, Val loss X ORN = {loss_XO_avg:.4f}')
#                 # print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}, Val loss Y ORN = {loss_YO_avg:.4f}')
#                 # print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}, Val loss Z ORN = {loss_ZO_avg:.4f}')
#                 # print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}, Val loss W ORN = {loss_WO_avg:.4f}')

#         writer.add_scalar("Loss/norm_val", total_recursive_val_loss / len(test_set), epoch)
#         writer.add_scalar("Loss/recursive_val", total_norm_val_loss / len(test_set), epoch)
#         writer.add_scalar("Loss/val_XP", loss_XP_avg, epoch)
#         # writer.add_scalar("Loss/val_YP", loss_YP_avg, epoch)
#         # writer.add_scalar("Loss/val_ZP", loss_ZP_avg, epoch)
#         # writer.add_scalar("Loss/val_XO", loss_XO_avg, epoch)
#         # writer.add_scalar("Loss/val_YO", loss_YO_avg, epoch)
#         # writer.add_scalar("Loss/val_ZO", loss_ZO_avg, epoch)
#         # writer.add_scalar("Loss/val_WO", loss_WO_avg, epoch)
# #
# # norm_path = norm_path[1:, :]
# # recursive_path = recursive_path[1:, :]
# # ground_truth = ground_truth[1:, :]
# # # print(norm_path.shape)
# # # print(norm_path[0:10, 6])
# # # print(recursive_path.shape)
# # # print(ground_truth.shape)
# # # print(ground_truth[0:10, 6])
# # total_samples = np.arange(3000)
# # plt.xlim(0, 3000)
# # plt.xlabel('Steps [n]')
# # plt.ylim(-0.25, 1.2)
# # metric = 3
# # label_list = ['X Position', 'Y Position', 'Z Position', 'X Orientation', 'Y Orientation', 'Z Orientation', 'W Orientation']
# # plt.ylabel(f'{label_list[metric]}')
# # plt.plot(total_samples, norm_path[:, metric], 'r-', label='One-step')
# # plt.plot(total_samples, recursive_path[:, metric], 'b-', label='Recursive')
# # plt.plot(total_samples, ground_truth[:, metric], 'g--', label='Ground Truth')
# # plt.legend()
# # plt.show()
# # writer.flush()
# # writer.close()