import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT, XML_DIR
from constants import A1_GRIPPER_POSITION_UNNORMALIZE_FN

from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import A1_GRIPPER_POSITION_NORMALIZE_FN
from constants import A1_GRIPPER_VELOCITY_NORMALIZE_FN


import IPython

e = IPython.embed

BOX_POSE = [None]  # to be changed from outside


def make_sim_env():
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
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
    xml_path = os.path.join(XML_DIR, f"a1_pick_n_place_cube.xml")
    physics = mujoco.Physics.from_xml_path(xml_path)
    task = PickNPlaceTask(random=False)
    env = control.Environment(
        physics,
        task,
        time_limit=20,
        control_timestep=DT,
        n_sub_steps=None,
        flat_observation=False,
    )
    return env


class SingleA1Task(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        right_arm_action = action[:6]
        normalized_gripper_action = action[6]

        gripper_action = A1_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_gripper_action)

        full_gripper_action = [gripper_action, -gripper_action]

        env_action = np.concatenate([right_arm_action, full_gripper_action])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        right_qpos_raw = qpos_raw[:8]
        right_arm_qpos = right_qpos_raw[:6]
        right_gripper_qpos = [A1_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        right_qvel_raw = qvel_raw[:8]
        right_arm_qvel = right_qvel_raw[:6]
        right_gripper_qvel = [A1_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = dict()
        obs["images"]["top"] = physics.render(height=480, width=640, camera_id="top")
        obs["images"]["angle"] = physics.render(
            height=480, width=640, camera_id="angle"
        )

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class PickNPlaceTask(SingleA1Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 3

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position

        # with physics.reset_context():
        #     physics.named.data.qpos[:8] = START_ARM_POSE_A1_SINGLE
        #     np.copyto(physics.data.ctrl, START_ARM_POSE_A1_SINGLE)
        #     assert BOX_POSE[0] is not None
        #     physics.named.data.qpos[8:8+7] = BOX_POSE[0]
        #     # print(f"{BOX_POSE=}")
        assert BOX_POSE[0] is not None
        physics.named.data.qpos[8 : 8 + 7] = BOX_POSE[0]

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[8 : 8 + 7]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
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


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(7)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7 : 7 + 6] = master_bot_right.dxl.joint_states.position[:6]
    # gripper action
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7 + 6] = normalized_right_pos
    return action


def test_sim_teleop():
    """Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work."""
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    master_bot_left = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name="gripper",
        robot_name=f"master_left",
        init_node=True,
    )
    master_bot_right = InterbotixManipulatorXS(
        robot_model="wx250s",
        group_name="arm",
        gripper_name="gripper",
        robot_name=f"master_right",
        init_node=False,
    )

    # setup the environment
    env = make_sim_env()
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation["images"]["angle"])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation["images"]["angle"])
        plt.pause(0.02)


if __name__ == "__main__":
    test_sim_teleop()
