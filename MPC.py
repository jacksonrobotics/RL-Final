import json
import numpy as np
import pybullet as p
import math
import yaml
import torch
import sys
import datetime
import torch.nn as nn
from pybullet_utils import bullet_client as bc
from crawler import CrawlerRobot
from self_model import QuickNN
np.set_printoptions(threshold=sys.maxsize)

"""
Collect the state and initialize target
Randomly sample actions [100 12xH vectors of actions] [ H is the number of steps per trajectory (H=10)]
Evaluate each vector based on cost function [cost function is how close it gets to the target point]
Take top 10% of best actions
Shift the mean and variance of the distribution that actions are drawn from to the mean and variance of the elites
[Take 1st step of the best action sequence]
Repeat for T iterations until the distribution converges?

"""

class MPC_CEM:

    def __init__(self, gui_enabled=True):

        if gui_enabled:
            self.bullet_client = bc.BulletClient(connection_mode=p.GUI)
        else:
            self.bullet_client = bc.BulletClient(connection_mode=p.DIRECT)

        self.robot = CrawlerRobot()
        self.robot.load()
        self.model = QuickNN()
        self.model.load_state_dict(torch.load('Memory_Dynamics_Self_Model_seed_8433'))
        self._lower_limit, self._upper_limit = self.robot.getJointLimits()  # These need to be in 12x1 Vector Shape
        self._lower_limit = np.asarray(self._lower_limit).reshape(-1, 1)
        self._upper_limit = np.asarray(self._upper_limit).reshape(-1, 1)
        self._N = 100  # Number of Random Samples drawn
        self._H = 10  # Number of steps taken in each action sequence
        self._predicted_States = np.empty((self._N, 7, self._H))
        self._predicted_Cost = np.empty((self._N, self._H))
        self._target_state = np.array(([2], [2], [0.25]))
        self._cutoff = 0.1
        self._action_samples = np.empty((self._N, 12, self._H))  # Structure is Trajectory_(Samples) x MOTORS x STEPS
        self._bounds = self.set_one(np.concatenate([self._upper_limit, self._lower_limit], 1))  # 12x2 Bounds
        self._lowest_cost = []
        self._best_mean = []
        self._best_var = []

        self._reset()

    def _reset(self):
        """Resets the environment and data containers."""
        self._n_steps = 0
        self._action = []
        self._seed = []
        self._observation = []
        self._current_position = []
        self._current_orientation = []
        self._n_1_state = []
        self._n_2_state = []
        self._n_3_state = []
        self._n_4_state = []
        self._n_5_state = []
        self._delta_position = []  # Delta Movement/Distance
        self._delta_orientation = []  # Delta Orientation
        self._best_mean = []
        self._best_var = []

    def step(self):

        self.bullet_client.stepSimulation()

        current_position = self.robot.getBodyPosition().reshape(-1, 1)
        current_orientation = self.robot.getBodyOrientation().reshape(-1, 1)

        self._n_steps += 1

        return current_position, current_orientation

    def actionDistribution(self, mean, variance):
        # Action Samples are drawn for each motor (TRAJECTORY_(SAMPLES) x MOTOR_COMMAND x STEPS)
        # Samples are drawn for H Amount of steps
        for sample_idx in range(self._action_samples.shape[0]):  # for i in 15
            for motor_idx in range(self._action_samples.shape[1]):  # for i in 12
                self._action_samples[sample_idx, motor_idx, :] = self.truncNorm(mean[motor_idx], variance[motor_idx], self._bounds)
        return self._action_samples  # Sample Shape is confirmed to have N x Actions x 1

    def truncNorm(self, mean, var, bounds):
        s = np.random.normal(mean, var, self._H)
        for i in range(s.shape[0]):
            if bounds[i][0] < s[i]:
                s[i] = bounds[i][0]
            elif bounds[i][1] > s[i]:
                s[i] = bounds[i][1]
        return s

    def set_one(self, array):
        array[:, 0] = 1
        array[:, 1] = -1
        return array

    def inputConcat(self, action, curr_position, curr_orientation, n_1_state, n_2_state, n_3_state, n_4_state, n_5_state):
        concat_input = np.transpose(np.concatenate((action, curr_position, curr_orientation, n_1_state, n_2_state, n_3_state, n_4_state, n_5_state), axis=0))
        tensor_input = torch.from_numpy(concat_input.astype(np.float32))
        return tensor_input  # Inputs probably need to be normalized

    def modelPrediction(self, full_actions, current_pos, current_ori, n_1_state, n_2_state, n_3_state, n_4_state, n_5_state):

        for i in range(full_actions.shape[0]):  # RUNS THROUGH EACH TRAJECTORY (N=15)
            input_position = current_pos
            input_orientation = current_ori
            n_1 = n_1_state
            n_2 = n_2_state
            n_3 = n_3_state
            n_4 = n_4_state
            n_5 = n_5_state
            for s in range(full_actions.shape[2]):  # RUNS THROUGH EACH STEP (H=10)
                action = np.reshape(full_actions[i, :, s], (12, 1))  # All 12 motor_commands at Trajectory i, Step s
                model_input = self.inputConcat(action, input_position, input_orientation, n_1, n_2, n_3, n_4, n_5)  # Reshape inputs for the model
                next_step = torch.flatten(torch.transpose(self.model(model_input), 0, 1))  # Determine next step
                self._predicted_States[i, :, s] = next_step.detach().numpy()
                n_5 = n_4
                n_4 = n_3
                n_3 = n_2
                n_2 = n_1
                n_1 = np.concatenate((input_position, current_orientation))
                input_position = np.reshape(self._predicted_States[i, 0:3, s], (3, 1))
                input_orientation = np.reshape(self._predicted_States[i, 3:7, s], (4, 1))

                _ = self.costFunction(i, s, input_position)
        return self._predicted_States

    def costFunction(self, traj_idx, step_idx, input_pos):
        # distance = sqrt((x_o - x_t)^2 + (y_o - y_t)^2 + (z_o - z_t)^2)
        cost = math.sqrt(np.sum(np.square((input_pos - self._target_state))))
        self._predicted_Cost[traj_idx, step_idx] = cost
        return self._predicted_Cost

    def findElites(self):
        # Only need to return the indices of the elites
        summed_cost = np.reshape(np.sum(self._predicted_Cost, 1), (self._N, 1))  # Should be same dim as N
        sorted_indices = np.argsort(summed_cost, 0)  # Sorts from smallest to largest
        cutoff = int(np.ceil(self._N * self._cutoff))  # Gets Number of Elites
        elites_indices = sorted_indices[0:cutoff]  # Gets the Indicies of the Elites
        lowest_cost = summed_cost[elites_indices[0]]
        average_cost = np.mean(summed_cost[elites_indices])

        return elites_indices, lowest_cost, average_cost
    def compareActions(self, current_position, average_cost):
        current_cost = math.sqrt(np.sum(np.square((current_position - self._target_state))))

        if current_cost < average_cost:  # If Current position is better than averaged positions from sampled actions
            return True
        else:
            return False


    def collect(self, action, current_position, current_orientation, seed, cost, mean, variance):
        """Collects data from a given step."""

        self._action.append(action.tolist())
        self._lowest_cost.append(cost.tolist())
        self._seed.append(seed)
        self._current_position.append(current_position.tolist())
        self._current_orientation.append(current_orientation.tolist())
        self._best_mean.append(mean.tolist())
        self._best_var.append(variance.tolist())


        return (self._action, self._current_position, self._current_orientation, self._seed)


    def save(self, steps):
        """Saves the collected data to a JSON file."""
        file_marker = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        filename = "MPC_{}_movement_data_{}.json".format(steps, file_marker)

        with open(filename, "w") as f:
            data = {
                # "observation": self._observation,
                "seed": self._seed,
                "action": self._action,
                "current position": self._current_position,
                "current orientation": self._current_orientation,
                "cost": self._lowest_cost,
                "mean": self._best_mean,
                "variance": self._best_var
            }
            json.dump(data, f, indent=4)


if __name__ == "__main__":

    with open('MPC.yaml', 'r') as file:
        config = yaml.safe_load(file)

    seed_gen = np.random.randint(low=0, high=10000)
    np.random.seed(seed_gen)
    torch.manual_seed(seed_gen)
    env = MPC_CEM(gui_enabled=config['gui_status'])
    sample_mean = np.zeros([12, 1])  # Sampling Mean for initial Sampling
    sample_variance = np.zeros([12, 1])  # Sampling Variance for initial Sampling
    sample_variance[:, 0] = 0.3  # Initial variance is 0.1
    cycle = 3
    best_traj_num = int(np.ceil(env._N * env._cutoff))
    best_actions = np.empty([best_traj_num, 12, env._H])
    tracking_best_cost = np.empty((cycle, 1))
    n_1_state = np.zeros((7, 1))
    n_2_state = np.zeros((7, 1))
    n_3_state = np.zeros((7, 1))
    n_4_state = np.zeros((7, 1))
    n_5_state = np.zeros((7, 1))

    for i in range(config['N']):
    # for i in range(5):
        print(i)
        current_position, current_orientation = env.step()  # CHECKED -> Prints out the current position

        for i in range(cycle):

            actions = env.actionDistribution(sample_mean, sample_variance)  # CHECKED -> Prints out (N x 12 x H), randomly generated array, within the allowable action limits
            predictions = env.modelPrediction(actions, current_position, current_orientation, n_1_state, n_2_state, n_3_state, n_4_state, n_5_state)  # CHECKED
            elites_indices, tracking_best_cost[i], average_elites_cost = env.findElites()  # CHECKED -> prints out the indices of the lowest Costs
            n_5_state = n_4_state
            n_4_state = n_3_state
            n_3_state = n_2_state
            n_2_state = n_1_state
            n_1_state = np.concatenate((current_position, current_orientation))

            for idx in range(elites_indices.shape[0]):
                best_actions[idx, :, :] = actions[idx, :, :]
            # if env.compareActions(current_position, average_elites_cost):  # if it returns True (average action leads away from target):
            #     pass  # Action Sampling distribution is not changed
            # else:
            #     sample_mean = np.reshape(np.mean(best_actions, (0, 2)), (12, 1))
            #     sample_variance = np.reshape(np.var(best_actions, (0, 2)), (12, 1))

        sample_mean = np.reshape(np.mean(best_actions, (0, 2)), (12, 1))
        sample_variance = np.reshape(np.var(best_actions, (0, 2)), (12, 1))

        take_action = best_actions[0, :, 0]

        env.robot.takeAction(take_action)
        env.collect(take_action, current_position, current_orientation, seed_gen, min(tracking_best_cost), sample_mean, sample_variance)

    if env._n_steps == config['N']:
        env.save(config['N'])

