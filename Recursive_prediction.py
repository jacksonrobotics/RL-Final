import json
import numpy as np
import pybullet
import math
import yaml
import torch
import torch.nn as nn
from pybullet_utils import bullet_client as bc
from crawler import CrawlerRobot
import matplotlib.pyplot as plt
from self_model import QuickNN

data_file='10000_random_movement_data_2023_11_08_08_26.json'
data_file = open(data_file)
data_load = json.load(data_file)
actions_data = 'action'
actions_data = np.transpose(np.asarray(data_load[actions_data]))


original_Pos = np.asarray([[0.0], [0.0], [0.2]])
original_Orn = np.asarray([[0.0], [0.0], [0.0], [1.0]])

model = QuickNN()
model.load_state_dict(torch.load('Dynamics_Self_Model_seed_2901'))

predictions = np.zeros((7, 1000))


input_state = np.concatenate((original_Pos, original_Orn), axis=0)
steps = np.arange(1000)

for s in range(1000):
    action = actions_data[:, s+2000]
    action = np.reshape(action, (12, 1))

    model_input = torch.from_numpy(np.transpose(np.concatenate((action, input_state),  axis=0)).astype(np.float32))  # Reshape inputs for the model
    next_step = torch.flatten(torch.transpose(model(model_input), 0, 1))  # Determine next step
    predictions[:, s] = next_step.detach().numpy()
    input_state = np.reshape(predictions[:, s], (7, 1))

predictions = np.transpose(predictions)
print(predictions.shape)
plt.xlim(np.min(predictions[:, 0], axis=0), np.max(predictions[:, 0], axis=0))
plt.xlabel('X [m]')
plt.ylim(np.min(predictions[:, 1], axis=0), np.max(predictions[:, 1], axis=0))
plt.ylabel('Y [m]')
plt.title('Recursive Predictions ' f'{max(steps)+1}' ' Steps')
plt.plot(predictions[:, 0], predictions[:, 1], 'b-', label='Trajectory 3')
plt.legend()
plt.show()


filename = "1000_random_movement_data_2023_11_08_08_26_Recursive_Predictions"

with open(filename, "w") as f:
    data = {
        "X predictions_1": list(predictions[0:1000, 0]),
        "Y predictions_1": list(predictions[0:1000, 1]),
            }
    json.dump(data, f, indent=4)


