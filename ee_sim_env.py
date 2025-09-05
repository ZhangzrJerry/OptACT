import numpy as np
import collections
import os


from constants import DT, XML_DIR, START_ARM_POSE_A1_SINGLE
from constants import A1_GRIPPER_POSITION_CLOSE
from constants import A1_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import A1_GRIPPER_POSITION_NORMALIZE_FN
from constants import A1_GRIPPER_VELOCITY_NORMALIZE_FN

from utils import sample_box_pose, sample_insertion_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import IPython

e = IPython.embed


def make_ee_sim_env():
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """

    xml_path = os.path.join(XML_DIR, f"a1_ee_pick_n_place_cube.xml")
    physics = mujoco.Physics.from_xml_path(xml_path)
    task = PickNPlaceEETask(random=False)
    env = control.Environment(
        physics,
        task,
        time_limit=20,
        control_timestep=DT,
        n_sub_steps=None,
        flat_observation=False,
    )
    return env


class SingleA1EETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        # set mocap position and quat
        # right
        np.copyto(physics.data.mocap_pos[0], action[:3])
        np.copyto(physics.data.mocap_quat[0], action[3:7])

        # set gripper
        g_ctrl = A1_GRIPPER_POSITION_UNNORMALIZE_FN(action[7])
        np.copyto(physics.data.ctrl, np.array([g_ctrl, -g_ctrl]))
        # print(f"{g_ctrl=} and {physics.data.ctrl=}")

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:8] = START_ARM_POSE_A1_SINGLE

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(
            physics.data.mocap_pos[0], np.array([0.31718881, 0.49999888, 0.29525084])
        )
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array(
            [A1_GRIPPER_POSITION_CLOSE, -A1_GRIPPER_POSITION_CLOSE]
        )
        np.copyto(physics.data.ctrl, close_gripper_control)
        # print(f"{physics.data.ctrl=}")

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        arm_qpos = qpos_raw[:6]
        gripper_qpos = [A1_GRIPPER_POSITION_NORMALIZE_FN(qpos_raw[6])]
        return np.concatenate([arm_qpos, gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        arm_qvel = qvel_raw[:6]
        gripper_qvel = [A1_GRIPPER_VELOCITY_NORMALIZE_FN(qvel_raw[6])]
        return np.concatenate([arm_qvel, gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = dict()
        obs["images"]["top"] = physics.render(height=480, width=640, camera_id="top")
        obs["images"]["angle"] = physics.render(
            height=480, width=640, camera_id="angle"
        )
        obs["images"]["front"] = physics.render(
            height=480, width=640, camera_id="front"
        )
        # used in scripted policy to obtain starting pose
        obs["mocap_pose"] = np.concatenate(
            [physics.data.mocap_pos[0], physics.data.mocap_quat[0]]
        ).copy()

        # used when replaying joint trajectory
        obs["gripper_ctrl"] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class PickNPlaceEETask(SingleA1EETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 3

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # self.initialize_robots(physics)
        # randomize box position
        box_pose = sample_box_pose()
        box_start_idx = physics.model.name2id("red_box_joint", "joint")
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], box_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[8 : 8 + 7]
        return env_state

    def get_reward(self, physics):
        # return whether gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_gripper = (
            "red_box",
            "a1_right/a1_8_gripper_finger_touch_right",
        ) in all_contact_pairs
        placed_on_plate = ("blue_plate", "red_box") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_gripper:  # attempted grasp
            reward = 1
        if touch_gripper and not touch_table:  # successful grasp
            reward = 2
        if placed_on_plate and not touch_gripper:  # successful placement
            reward = 3

        # print(f"{reward=}")
        return reward
