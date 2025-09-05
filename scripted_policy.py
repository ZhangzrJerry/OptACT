import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env

import IPython
import cv2

e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint["xyz"]
        curr_quat = curr_waypoint["quat"]
        curr_grip = curr_waypoint["gripper"]
        next_xyz = next_waypoint["xyz"]
        next_quat = next_waypoint["quat"]
        next_grip = next_waypoint["gripper"]
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]["t"] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]["t"] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(
            self.curr_left_waypoint, next_left_waypoint, self.step_count
        )
        right_xyz, right_quat, right_gripper = self.interpolate(
            self.curr_right_waypoint, next_right_waypoint, self.step_count
        )

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class PickAndPlacePolicy(BasePolicy):

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain right waypoints
        if self.right_trajectory[0]["t"] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        right_xyz, right_quat, right_gripper = self.interpolate(
            self.curr_right_waypoint, next_right_waypoint, self.step_count
        )

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_right])

    def generate_trajectory(self, ts_first):
        init_mocap_pose = ts_first.observation["mocap_pose"]

        box_info = np.array(ts_first.observation["env_state"])
        # print(box_info)
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose[3:])
        # gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([-0.3, 0.5, 0.25])

        box_offset = -0.12
        plate_offset = -0.2
        plate_xyz = np.array([-0.17, 0.5, 0.002])

        self.right_trajectory = [
            {
                "t": 0,
                "xyz": init_mocap_pose[:3],
                "quat": init_mocap_pose[3:],
                "gripper": 0,
            },  # sleep
            {
                "t": 90,
                "xyz": box_xyz + np.array([box_offset, 0, 0.14]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # approach the cube
            {
                "t": 120,
                "xyz": box_xyz + np.array([box_offset, 0, 0.04]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # go down
            {
                "t": 150,
                "xyz": box_xyz + np.array([box_offset, 0, 0.04]),
                "quat": gripper_pick_quat.elements,
                "gripper": 0,
            },  # grip the cube
            {
                "t": 250,
                "xyz": plate_xyz + np.array([plate_offset, 0, 0.2]),
                "quat": gripper_pick_quat.elements,
                "gripper": 0,
            },  # drag to plate xy
            {
                "t": 300,
                "xyz": plate_xyz + np.array([plate_offset, 0, 0.2]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # drag to plate z
            {
                "t": 350,
                "xyz": plate_xyz + np.array([plate_offset, 0, 0.2]),
                "quat": gripper_pick_quat.elements,
                "gripper": 1,
            },  # stay
            # {"t": 350, "xyz": box_xyz + np.array([box_offset, 0, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 1}, # back to init
            # {"t": 400, "xyz": box_xyz + np.array([box_offset, 0, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        ]


def test_policy():
    # Example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # Setup the environment
    episode_len = SIM_TASK_CONFIGS["episode_len"]
    env = make_ee_sim_env()

    # Video settings
    video_filename = f"simulation.mp4"
    fps = 20
    frame_width = 640  # Adjust based on your image size
    frame_height = 480
    video_writer = cv2.VideoWriter(
        video_filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation["images"]["angle"])
            plt.ion()

        policy = PickAndPlacePolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)

            # Render image and update the plot (if onscreen_render is True)
            img_rgb = ts.observation["images"]["angle"]
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(img_bgr)

            if onscreen_render:
                plt_img.set_data(img_rgb)
                plt.pause(0.02)

        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

    # Release the video writer and save the video
    video_writer.release()
    print(f"Video saved as {video_filename}")


if __name__ == "__main__":
    test_policy()
