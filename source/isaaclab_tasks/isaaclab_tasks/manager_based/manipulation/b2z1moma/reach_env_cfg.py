# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfDiscreteObstaclesTerrainCfg

from . import mdp
import random


##
# Scene definition
##

@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""
    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.6)),
        spawn=GroundPlaneCfg(),
    )

    terrain_cfg = TerrainImporterCfg(
        num_envs=2,
        env_spacing=10,
        prim_path="/World/Ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(10, 10),
            border_width=5.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.1,
            slope_threshold=0.75,
            color_scheme="height",
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    horizontal_scale=0.1,
                    vertical_scale=0.1,
                    border_width=0.0,
                    num_obstacles=60,
                    obstacle_height_mode="choice",
                    obstacle_width_range=(0.4, 1.1),
                    obstacle_height_range=(0.9, 3.5),
                    platform_width=0.0,
                ),
            },
        ),
        debug_vis=True,
    )

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1, 0, -0.3), rot=(1, 0, 0, 0)),
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(0.3 + random.random() * 0.2, 0.3 + random.random() * 0.2, 0.3 + random.random() * 1.2),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ) for i in range(100)
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=4, max_angular_velocity=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # robots
    robot: ArticulationCfg = MISSING

    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/lidar_link",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.0)),
        mesh_prim_paths=["/World/Ground"],
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=8, vertical_fov_range=[-30, 30], horizontal_fov_range=[-120, 120], horizontal_res=1.0
        ),
        debug_vis=True,
        max_distance=5.0
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # ee_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name=MISSING,
    #     resampling_time_range=(4.0, 4.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.35, 0.65),
    #         pos_y=(-0.2, 0.2),
    #         pos_z=(0.15, 0.5),
    #         roll=(0.0, 0.0),
    #         pitch=MISSING,  # depends on end-effector axis
    #         yaw=(-3.14, 3.14),
    #     ),
    # )

    robot_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="trunk",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(1.5, 2.0),
            pos_y=(-0.2, 0.2),
            pos_z=(0.2, 0.2),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),  # depends on end-effector axis
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm | None = None
    base_action: ActionTerm | None = None
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["FR_.*", "FL_.*", "joint[1-6]", "joint_.*"]),
            }
        )

        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))

        robot_pos_err = ObsTerm(
            func=mdp.robot_pos_err,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                "command_name": "robot_pose",
            }
        )

        # robot_position = ObsTerm(
        #     func=mdp.root_pos_w,
        #     noise=Unoise(n_min=-0.01, n_max=0.01),
        #     params={
        #     "robot_cfg": SceneEntityCfg("robot", body_names="trunk")}
        # )

        cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names="trunk")}
        )

        # pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        # robot_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "robot_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_.*"]),
        },
    )

    reset_robot_arm = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-1.57, -1.57),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"]),
        },
    )

    reset_robot_leg_hip = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip.*"]),
        },
    )

    reset_robot_leg_thigh = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.6, 0.6),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh.*"]),
        },
    )

    reset_robot_leg_thigh = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-1.5, -1.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf.*"]),
        },
    )

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.2), "y": (-0.3, 0.3), "z": (-0.3, -0.2)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube", body_names="Cube"),
        },
    )

    reset_robot_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.0), "y": (-0.3, 0.3), "z": (-0.1, -0.1)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot", body_names="world"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    # end_effector_position_tracking = RewTerm(
    #     func=mdp.position_command_error,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    # )
    # end_effector_position_tracking_fine_grained = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.1, "command_name": "ee_pose"},
    # )
    # end_effector_orientation_tracking = RewTerm(
    #     func=mdp.orientation_command_error,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    # )

    # end_robot_position_tracking = RewTerm(
    #     func=mdp.position_command_error,
    #     weight=-0.02,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="trunk"), "command_name": "robot_pose"},
    # )
    # end_robot_position_tracking_fine_grained = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=0.01,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="trunk"), "std": 0.1, "command_name": "robot_pose"},
    # )

    robot_vel_tracking = RewTerm(
        func=mdp.robot_pos_command_error,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="trunk"), "command_name": "robot_pose"},
    )

    object_move = RewTerm(
        func=mdp.object_move_error,
        weight=0.1,
        params={"robot_cfg": SceneEntityCfg("robot", body_names="trunk"), "command_name": "robot_pose"},
    )

    # joint_pos_limit = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-0.001,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["FR_.*", "FL_.*"])},
    # )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["FR_.*", "FL_.*", "jointvel.*"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # object_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.0005, "num_steps": 20000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.0005, "num_steps": 20000}
    )

    # object_move = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "object_move", "weight": 0.04, "num_steps": 10000}
    # )


##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0