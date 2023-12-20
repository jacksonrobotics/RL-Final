import json
import datetime
import torch
import numpy as np
import pybullet as p
from pybullet_utils import bullet_client as bc
from crawler import CrawlerRobot
import yaml

current_time = datetime.datetime.now()

class DataCollection:

    def __init__(self, gui_enabled=True):

        if gui_enabled:
            self.bullet_client = bc.BulletClient(connection_mode=p.GUI)
        else:
            self.bullet_client = bc.BulletClient(connection_mode=p.DIRECT)

        self.robot = CrawlerRobot()
        self.robot.load()

        # set seed
        self._reset()

    def _reset(self):
        """Resets the environment and data containers."""
        self._action = []
        self._n_1_state = []
        self._n_2_state = []
        self._n_3_state = []
        self._n_4_state = []
        self._n_5_state = []
        self._seed = []
        self._n_steps = 0
        self._observation = []
        self._reward = np.array([])
        self._flag = np.array([])
        self._current_position = []
        self._current_orientation = []
        self._delta_position = []  # Delta Movement/Distance
        self._delta_orientation = []  # Delta Orientation

    def step(self):

        # RL things: all none for now except for observation
        reward, done, info = None, None, None

        # self.robot.randomMovement()
        self.bullet_client.stepSimulation()

        observation = self.robot.getObservation()
        delta_position = self.robot.getDeltaPosition()
        delta_orientation = self.robot.getDeltaOrientation()
        current_position = self.robot.getBodyPosition()
        current_orientation = self.robot.getBodyOrientation()


        self._n_steps += 1

        return observation, delta_position, delta_orientation, reward, done, info, current_position, current_orientation

    def collect(self, action, n_1_state, n_2_state, n_3_state, n_4_state, n_5_state, distance, orientation, obs, r, done, current_position, current_orientation, seed):
        """Collects data from a given step."""

        self._action.append(action.tolist())
        self._n_1_state.append(n_1_state.tolist())
        self._n_2_state.append(n_2_state.tolist())
        self._n_3_state.append(n_3_state.tolist())
        self._n_4_state.append(n_4_state.tolist())
        self._n_5_state.append(n_5_state.tolist())
        self._observation.append(obs)
        self._seed.append(seed)
        self._reward = np.append(self._reward, r) 
        self._flag = np.append(self._flag, done)  
        self._delta_position.append(distance.tolist())
        self._delta_orientation.append(orientation.tolist())
        self._current_position.append(current_position.tolist())
        self._current_orientation.append(current_orientation.tolist())

        return (self._action, self._delta_position, self._delta_orientation, self._observation,
                self._reward, self._flag, self._current_position, self._current_orientation)

    def save(self, steps):
        """Saves the collected data to a JSON file."""
        file_marker = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        filename = "{}_random_movement_data_{}.json".format(steps, file_marker)

        with open(filename, "w") as f:
            data = {
                # "observation": self._observation,
                "seed": self._seed,
                "action": self._action,
                "n_1_state": self._n_1_state,
                "n_2_state": self._n_2_state,
                "n_3_state": self._n_3_state,
                "n_4_state": self._n_4_state,
                "n_5_state": self._n_5_state,
                # "delta distance": self._delta_position,
                # "delta orientation": self._delta_orientation,
                "current position": self._current_position,
                "current orientation": self._current_orientation
            }
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    with open('data_collection.yaml', 'r') as file:
        config = yaml.safe_load(file)

    env = DataCollection(gui_enabled=config['gui_status'])
    step = []

    terminal_step = int(config['N'])*int(config['T'])

    ########################################################################################

    if not config['multiTrajectory']:

        seed_gen = np.random.randint(low=0, high=10000)
        np.random.seed(seed_gen)
        torch.manual_seed(seed_gen)
        n_1_state = np.zeros(7)
        n_2_state = np.zeros(7)
        n_3_state = np.zeros(7)
        n_4_state = np.zeros(7)
        n_5_state = np.zeros(7)

        for i in range(config['N']):
            obs, delta_position, delta_orientation, r, done, _, current_position, current_orientation = env.step()

            if config['actionType'] == 1:

                """Random Movement"""
                action = env.robot.randomMovement()
                env.collect(action, n_1_state, n_2_state, n_3_state, n_4_state, n_5_state, delta_position, delta_orientation, obs, r, done, current_position,
                            current_orientation, seed_gen)
                n_5_state = n_4_state
                n_4_state = n_3_state
                n_3_state = n_2_state
                n_2_state = n_1_state
                n_1_state = np.concatenate((current_position, current_orientation))

            elif config['actionType'] == 2:

                """Parameter-based Random Movement"""
                parameters = env.robot.random_para()
                action = env.robot.sin_move(parameters)
                env.collect(action, n_1_state, n_2_state, n_3_state, n_4_state, n_5_state, delta_position, delta_orientation, obs, r, done, current_position,
                            current_orientation, seed_gen)
                n_5_state = n_4_state
                n_4_state = n_3_state
                n_3_state = n_2_state
                n_2_state = n_1_state
                n_1_state = np.concatenate((current_position, current_orientation))

            if not config['gui_status']:
                pass
            else:
                p.resetDebugVisualizerCamera(cameraDistance=1.0,
                                             cameraYaw=180,
                                             cameraPitch=-45,
                                             cameraTargetPosition=p.getBasePositionAndOrientation(env.robot._crawlerId)[
                                                 0])

            if env._n_steps == config['N']:
                env.save(config['N'])
    else:
        ########################################################################################
        for x in range(config['T']):

            seed_gen = np.random.randint(low=0, high=10000)
            np.random.seed(seed_gen)
            torch.manual_seed(seed_gen)

            print("T: ", x)
            env.robot.reset()

            # SHOULD I USE ZEROS OR INITIAL POSITION
            # n_1_state = np.array([0, 0, 0.2, 0, 0, 0, 1])
            # n_2_state = np.array([0, 0, 0.2, 0, 0, 0, 1])
            # n_3_state = np.array([0, 0, 0.2, 0, 0, 0, 1])
            # n_4_state = np.array([0, 0, 0.2, 0, 0, 0, 1])
            # n_5_state = np.array([0, 0, 0.2, 0, 0, 0, 1])
            n_1_state = np.zeros(7)
            n_2_state = np.zeros(7)
            n_3_state = np.zeros(7)
            n_4_state = np.zeros(7)
            n_5_state = np.zeros(7)

            for i in range(config['N']):

                step = (int(x)+1) * (int(i)+1)
                obs, delta_position, delta_orientation, r, done, _, current_position, current_orientation = env.step()

                if i == 0:
                    current_position = np.array([0, 0, 0.2])
                    current_orientation = np.array([0, 0, 0, 1])


                if config['actionType'] == 1:

                    """Random Movement"""
                    action = env.robot.randomMovement()
                    env.collect(action, n_1_state, n_2_state, n_3_state, n_4_state, n_5_state, delta_position, delta_orientation, obs, r, done, current_position,
                                current_orientation, seed_gen)
                    n_5_state = n_4_state
                    n_4_state = n_3_state
                    n_3_state = n_2_state
                    n_2_state = n_1_state
                    n_1_state = np.concatenate((current_position, current_orientation))

                elif config['actionType'] == 2:

                    if i == 0:
                        current_position = np.array([0, 0, 0.2])
                        current_current_orientation = np.array([0, 0, 0, 1])

                    """Parameter-based Random Movement"""
                    parameters = env.robot.random_para()
                    action = env.robot.sin_move(parameters)
                    env.collect(action, n_1_state, n_2_state, n_3_state, n_4_state, n_5_state, delta_position, delta_orientation, obs, r, done, current_position,
                                current_orientation, seed_gen)
                    n_5_state = n_4_state
                    n_4_state = n_3_state
                    n_3_state = n_2_state
                    n_2_state = n_1_state
                    n_1_state = np.concatenate((current_position, current_orientation))

                if not config['gui_status']:
                    pass
                else:
                    p.resetDebugVisualizerCamera(cameraDistance=1.0,
                                                 cameraYaw=180,
                                                 cameraPitch=-45,
                                                 cameraTargetPosition=
                                                 p.getBasePositionAndOrientation(env.robot._crawlerId)[
                                                     0])

            if terminal_step == step:
                env.save(step)




