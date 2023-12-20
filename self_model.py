import json
import datetime
import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


class LSTMPredictor(nn.Module):
    def __init__(self, n_hidden=128):
        super(LSTMPredictor, self).__init__()
        self._n_hidden = n_hidden
        self.lstm1 = nn.LSTMCell(19, self._n_hidden)  # Input size is states?
        self.lstm2 = nn.LSTMCell(self._n_hidden, self._n_hidden)
        self.linear = nn.Linear(self._n_hidden, 7)


    def forward(self, x, future=0):
        outputs = []
        n_samples = x.size(0)

        h_t = torch.zeros(n_samples, self._n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self._n_hidden, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self._n_hidden, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self._n_hidden, dtype=torch.float32)

        for input_t in x.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        outputs = torch.cat(outputs, 1)
        return outputs



class QuickNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            # nn.Linear(19, 128),
            nn.Linear(54, 128),  # 19 + 7 + 7 + 7 + 7, for Actions, Current state, n-1, ... n-4, Previous states 
            # 19, 26, 33, 40, 47, 54, (61 for n+1)
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 7),
            nn.Tanh(),
        )

    def forward(self, x):
        y_predict = self.linear_relu_stack(x)
        return y_predict

class MovementDataset(Dataset):
    def __init__(self, data_file='20000_random_movement_data_2023_11_26_21_51.json',
                 actions_data='action',
                 n_1_data='n_1_state',
                 n_2_data='n_2_state',
                 n_3_data='n_3_state',
                 n_4_data='n_4_state',
                 n_5_data='n_5_state',
                 position_data='current position',
                 orientation_data='current orientation'):
        self._data_file = open(data_file)
        self._data_load = json.load(self._data_file)
        # Individual Category Arrays
        self._actions_data = self.getNorm(self._data_load[actions_data])  # (SAMPLE x 12)
        self._n_1_data = self.getNorm(self._data_load[n_1_data])
        self._n_2_data = self.getNorm(self._data_load[n_2_data])
        self._n_3_data = self.getNorm(self._data_load[n_3_data])
        self._n_4_data = self.getNorm(self._data_load[n_4_data])
        self._n_5_data = self.getNorm(self._data_load[n_5_data])
        self._position_data = self.getNorm(self._data_load[position_data])
        self._orientation_data = self.getNorm(self._data_load[orientation_data])
        self._DataSet = self.getDataSet(self._actions_data, self._position_data, self._orientation_data, self._n_1_data, self._n_2_data, self._n_3_data, self._n_4_data, self._n_5_data)
        # self._DataSet = self.getDataSet(self._actions_data, self._position_data, self._orientation_data, self._n_1_data, self._n_2_data, self._n_3_data, self._n_4_data, self._n_5_data)
        self._trainData, self._valData, self._testData = self.customSplit(self._DataSet)

        self.n_samples = len(self._DataSet)  # Only returns the number of SAMPLES for Actions

    def __getitem__(self, index):
        X = self._DataSet[index, 0:54] #########################################
        y = self._DataSet[index, 54:61]
        return X, y

    def __len__(self):
        # len(dataset)
        return self.n_samples

    # def getDataSet(self, dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6, dataset_7, dataset_8):
    def getDataSet(self, dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6, dataset_7, dataset_8):
        # Separate data into input and Output Arrays
        # input_data = np.concatenate((dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6, dataset_7, dataset_8), axis=1)  # (action_array, position_array, orientation_array)
        input_data = np.concatenate((dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6, dataset_7, dataset_8), axis=1)
        output_data = np.concatenate((dataset_2, dataset_3), axis=1)  # (position_array, orientation_array) 7

        # Shift the data sets
        input_array = input_data[0:-1, :]
        output_array = output_data[1:, :]

        # Combine Arrays into 1 Dataset
        complete_data = np.concatenate((input_array, output_array), axis=1)  

        pre_tensor = torch.from_numpy(complete_data.astype(np.float32))
        complete_tensor = pre_tensor.requires_grad_()
        return complete_tensor

    def getNorm(self, input):
        array = np.asarray(input)
        for x in range(array.shape[1]):
            array_max = np.max(array[:, x], axis=0)
            array_min = np.min(array[:, x], axis=0)
            array[:, x] = (2 * ((array[:, x] - array_min) / (array_max - array_min))) - 1
        return array
        # return array, array_max, array_min

    def customSplit(self, data):  # Data is the data set
        data_size = data.shape[0]  # Returns the NUMBER OF SAMPLES (Assuming Samples x Features)
        train_idx = int(np.trunc(0.8*data_size))
        val_idx = int(train_idx + np.ceil(0.1*data_size))
        test_idx = int(val_idx + np.ceil(0.1*data_size))
        train_data = data[0:train_idx, :]
        val_data = data[train_idx:val_idx, :]
        test_data = data[val_idx:, :]  # Can also do [:, val_idx:]
        # print(train_data.shape, val_data.shape, test_data.shape)
        return train_data, val_data, test_data

    # def saveScale(self):
    #     file_marker = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    #     filename = "scaling_{}.json".format(file_marker)
    #     with open(filename, "w") as f:
    #         data = {
    #             "action max": self._actions_max,
    #             "action min":  self._actions_min,
    #             "position max": self._position_max,
    #             "position min": self._position_min,
    #             "orientation max": self._orientation_max,
    #             "orientation min": self._orientation_min
    #         }
    #         json.dump(data, f, indent=4)

######################################################################################################################
# Initialize dataset Class


seed_gen = np.random.randint(low=0, high=10000)
np.random.seed(seed_gen)
torch.manual_seed(seed_gen)

crawler1_dataset = MovementDataset()

#Training and test datasets
custom_split = True  # Custom split uses 1st 80% of trajectory for training, and splits the rest for val and test

if custom_split == True:
    train_set = crawler1_dataset._trainData
    val_set = crawler1_dataset._valData
else:
    train_set, val_set = torch.utils.data.random_split(crawler1_dataset, [int(0.8 * len(crawler1_dataset)),
                                                                          len(crawler1_dataset) - int(
                                                                              0.8 * len(crawler1_dataset))])


batch_size = 4  # HYPERPARAMETER
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Initialize the model
model = QuickNN()

# Loss and Optimizer
learning_rate = 0.01  # HYPERPARAMETER
# w_decay = 0.005  # HYPERPARAMETER
criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=w_decay)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # try Adam next

# training
num_epochs = 40  # HYPERPARAMETER
total_samples = len(train_set)
n_iterations = math.ceil(total_samples / batch_size)
print(total_samples, n_iterations)

norm_path = np.empty((1, 7))
recursive_path = np.empty((1, 7))
ground_truth = np.empty((1, 7))
x_pos_diff = np.empty((1, 1))
Traj_len = 200  # Number of teps in each trajectory; from data_collection.yaml
n_prior_reset = Traj_len/batch_size  # Should equal (200/4) 50
# print(type())

if __name__ == "__main__":

    for epoch in range(num_epochs):
        total_train_loss = 0
        total_norm_val_loss = 0
        total_recursive_val_loss = 0
        loss_XP = 0
        loss_YP = 0
        loss_ZP = 0
        loss_XO = 0
        loss_YO = 0
        loss_ZO = 0
        loss_WO = 0
        mean_loss_XP = []
        mean_loss_YP = []
        mean_loss_ZP = []
        mean_loss_XO = []
        mean_loss_YO = []
        mean_loss_ZO = []
        mean_loss_WO = []

        for i, state_action_pairs in enumerate(train_dataloader):  # TRAINING

            if custom_split == True:
                state = state_action_pairs[:, 0:54]
                y_action = state_action_pairs[:, 54:61]
            else:
                state, y_action = state_action_pairs

            # zero gradients
            optimizer.zero_grad()

            y_predicted = model(state)
            loss = criterion(y_predicted, y_action)
            total_train_loss += loss

            # Compute Loss
            loss.backward(retain_graph=True)

            # Update Weights
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'iteration: {i + 1}, training loss = {loss.item():.4f}')
                print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_iterations}')
        # send to tensorboard
        writer.add_scalar("Loss/train", total_train_loss / len(train_set), epoch)

        with torch.no_grad():  # EVAL

            for i, state_action_pairs in enumerate(val_dataloader):

                if (i)%50 == 0:
                    previous_prediction = torch.zeros(7)  # []
                    n_1_prediction = torch.zeros(7)
                    n_2_prediction = torch.zeros(7)
                    n_3_prediction = torch.zeros(7)
                    n_4_prediction = torch.zeros(7)
                    n_5_prediction = torch.zeros(7)

                if custom_split == True:
                    state = state_action_pairs[:, 0:54]
                    y_val_action = state_action_pairs[:, 54:61]
                else:
                    state, y_val_action = state_action_pairs

                # Original Prediction pipeline
                norm_y_predicted = model(state)
                # Compute Loss
                norm_val_loss = criterion(norm_y_predicted, y_val_action)

                # Collect Predicted and Ground Truth Data
                norm_y_predicted_array = norm_y_predicted.numpy()  # Convert tensor to numpy array
                norm_path = np.append(norm_path, norm_y_predicted_array, axis=0)  # Collect new prediction
                ground_truth = np.append(ground_truth, state[:, 12:19], axis=0)  # Collect Ground Truth

                # Consolidate Loss
                total_norm_val_loss += norm_val_loss

                ###################################
                # The State input and the NEXT STEP are the SAME
                ###################################

                # Recursive Prediction Pipeline
                if i == 0:
                    recursive_y_predicted = model(state)

                else:
                    state[:, 12:19] = previous_prediction  # State = Action, Sn, Sn-1, Sn-2 
                    state[:, 19:26] = n_1_prediction
                    state[:, 26:33] = n_2_prediction
                    state[:, 33:40] = n_3_prediction
                    state[:, 40:47] = n_4_prediction
                    state[:, 47:54] = n_5_prediction
                    recursive_y_predicted = model(state)

                recursive_path = np.append(recursive_path, recursive_y_predicted, axis=0)
                recursive_val_loss = criterion(recursive_y_predicted, y_val_action)
                total_recursive_val_loss += recursive_val_loss
                n_5_prediction = n_4_prediction
                n_4_prediction = n_3_prediction
                n_3_prediction = n_2_prediction
                n_2_prediction = n_1_prediction
                n_1_prediction = previous_prediction
                previous_prediction = recursive_y_predicted


                #############################################################
                # Need to clear each variable before the loop
                loss_XP = 0
                loss_YP = 0
                loss_ZP = 0
                loss_XO = 0
                loss_YO = 0
                loss_ZO = 0
                loss_WO = 0

                for e in range(batch_size):
                    loss_0 = criterion(norm_y_predicted[e][0], y_val_action[e][0])
                    loss_1 = criterion(norm_y_predicted[e][1], y_val_action[e][1])
                    loss_2 = criterion(norm_y_predicted[e][2], y_val_action[e][2])
                    loss_3 = criterion(norm_y_predicted[e][3], y_val_action[e][3])
                    loss_4 = criterion(norm_y_predicted[e][4], y_val_action[e][4])
                    loss_5 = criterion(norm_y_predicted[e][5], y_val_action[e][5])
                    loss_6 = criterion(norm_y_predicted[e][6], y_val_action[e][6])
                    # Summing the loss for each prediction in batch
                    loss_XP += loss_0
                    loss_YP += loss_1
                    loss_ZP += loss_2
                    loss_XO += loss_3
                    loss_YO += loss_4
                    loss_ZO += loss_5
                    loss_WO += loss_6
                # Averaging the loss over num_samples per batch
                loss_XP_avg = loss_XP / batch_size
                loss_YP_avg = loss_YP / batch_size
                loss_ZP_avg = loss_ZP / batch_size
                loss_XO_avg = loss_XO / batch_size
                loss_YO_avg = loss_YO / batch_size
                loss_ZO_avg = loss_ZO / batch_size
                loss_WO_avg = loss_WO / batch_size
                #############################################################

            # send to tensorboard
                if (i + 1) % 10 == 0:
                    print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}, Validation loss = {norm_val_loss.item():.4f}')
                    # print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}, Val loss X POS = {loss_XP_avg:.4f}')
                    # print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}, Val loss Y POS = {loss_YP_avg:.4f}')
                    # print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}, Val loss Z POS = {loss_ZP_avg:.4f}')
                    # print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}, Val loss X ORN = {loss_XO_avg:.4f}')
                    # print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}, Val loss Y ORN = {loss_YO_avg:.4f}')
                    # print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}, Val loss Z ORN = {loss_ZO_avg:.4f}')
                    # print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}, Val loss W ORN = {loss_WO_avg:.4f}')

            writer.add_scalar("Loss/norm_val", total_norm_val_loss / len(val_set), epoch)
            writer.add_scalar("Loss/recursive_val", total_recursive_val_loss / len(val_set), epoch)
            writer.add_scalar("Loss/val_XP", loss_XP_avg, epoch)  # Average losses for Val Loss
            writer.add_scalar("Loss/val_YP", loss_YP_avg, epoch)
            writer.add_scalar("Loss/val_ZP", loss_ZP_avg, epoch)
            writer.add_scalar("Loss/val_XO", loss_XO_avg, epoch)
            writer.add_scalar("Loss/val_YO", loss_YO_avg, epoch)
            writer.add_scalar("Loss/val_ZO", loss_ZO_avg, epoch)
            writer.add_scalar("Loss/val_WO", loss_WO_avg, epoch)

    torch.save(model.state_dict(), f'Memory_Dynamics_Self_Model_seed_{seed_gen}')
    #
    # norm_path = norm_path[1:, :]
    # recursive_path = recursive_path[1:, :]
    # ground_truth = ground_truth[1:, :]
    # x_pos_diff = x_pos_diff[1:, :]
    # print(ground_truth.shape)
    # # print(norm_path[0:10, 6])
    # # print(recursive_path.shape)
    # total_samples = np.arange(40000)
    # plt.xlim(19900, 20000)
    # plt.xlabel('Steps [n]')
    # metric = 0
    # plt.ylim(np.min(ground_truth[:, metric], axis=0), np.max(ground_truth[:, metric], axis=0))
    # label_list = ['X Position', 'Y Position', 'Z Position', 'X Orientation', 'Y Orientation', 'Z Orientation', 'W Orientation']
    # plt.ylabel(f'{label_list[metric]}')
    # plt.title('Seed: 0, Test 2')
    # plt.plot(total_samples, norm_path[:, metric], 'r-', label='One-step')
    # # plt.plot(total_samples, recursive_path[:, metric], 'g-', label='Recursive')
    # plt.plot(total_samples, ground_truth[:, metric], 'b-', label='Ground Truth')
    # plt.legend()
    # plt.show()
    writer.flush()
    writer.close()
