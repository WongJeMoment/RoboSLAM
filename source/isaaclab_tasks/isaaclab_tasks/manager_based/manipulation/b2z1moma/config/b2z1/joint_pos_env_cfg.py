# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Reach-b2z1-v0 --num_envs 2048 --headless
# python scripts/reinforcement_learning/skrl/play.py --task=Isaac-Reach-b2z1-Play-v0 --num_envs 16
import math

from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

import isaaclab_tasks.manager_based.manipulation.b2z1moma.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.b2z1moma.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from .b2z1_model import UNITREE_B2Z1_CFG # isort: skip
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip

##
# Environment configuration
##


@configclass
class b2z1ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = UNITREE_B2Z1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        # self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link06"]
        # self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link06"]
        # self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link06"]

        # override actions "joint[1-6]", "joint_.*", "FR_.*", "FL_.*"
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["FR_.*", "FL_.*", "joint[1-4]"], 
            use_default_offset=False,
            clip={
                ".*_hip.*":  (-0.3, 0.5),
                ".*_thigh.*":(0.4, 0.6),
                ".*_calf.*": (-1.3, -1.5),
                # "joint_.*": (0.0, 0.0),
                "joint1": (-2.0, -1.0),
                "joint2": (2.0,2.0),
                "joint3": (-1.5,-1.5),
                "joint4": (-0.54,-0.54),
                # "joint5": (0.0,0.0),
                # "joint6": (0.0,0.0),
            },
            offset={
                "joint1":-1.57,
            }
        )

        self.actions.base_action = mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=["jointvel.*"], use_default_offset=True, clip={"jointvel.*":(-0.3,0.3)}
        )
        #self.actions.base_action = mdp.JointVelocityActionCfg()
        # override command generator body
        # end-effector is along z-direction
        # self.commands.ee_pose.body_name = "link06"
        # self.commands.ee_pose.ranges.pitch = (-3.14, 3.14)
        # self.commands.ee_pose.ranges.roll = (-3.14, 3.14)
        # self.commands.ee_pose.ranges.yaw = (-3.14, 3.14)
        # self.commands.ee_pose.ranges.pos_x = (0.0, 0.0)
        # self.commands.ee_pose.ranges.pos_y = (0.2, 0.5)
        # self.commands.ee_pose.ranges.pos_z = (0.1, 0.5)


@configclass
class b2z1ReachEnvCfg_PLAY(b2z1ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
