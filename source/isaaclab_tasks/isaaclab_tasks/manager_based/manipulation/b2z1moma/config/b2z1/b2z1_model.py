import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

usd_path = "/home/wongje/IsaacLab_remove_rl/IsaacLab/robot/b2_z1_lidar_fleg.usd"
UNITREE_B2Z1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0, fix_root_link=True
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint_x": 0.0,
            "joint_y": 0.0,
            "joint_z": 0.0,
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "jointGripper": 0.0,
            ".*_hip.*":  0.0,
            ".*_thigh.*":0.67,
            ".*_calf.*": -1.3,
        },
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "base_vel": ImplicitActuatorCfg(
            joint_names_expr=["jointvel.*"],
            effort_limit=500.0,
            velocity_limit=0.5,
            stiffness=0.0,
            damping=500.0,
        ),
        "float_base": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],
            effort_limit=500.0,
            velocity_limit=3.14,
            stiffness=10000.0,
            damping=500.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit=87.0,
            velocity_limit=3.14,
            stiffness=800.0,
            damping=4.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["jointGripper"],
            effort_limit=800.0,
            velocity_limit=3.14,
            stiffness=800.0,
            damping=4.0,
        ),
        "fl": ImplicitActuatorCfg(
            joint_names_expr=["FL.*"],
            effort_limit=800.0,
            velocity_limit=3.14,
            stiffness=800.0,
            damping=4.0,
        ),
        "fr": ImplicitActuatorCfg(
            joint_names_expr=["FR.*"],
            effort_limit=800.0,
            velocity_limit=3.14,
            stiffness=800.0,
            damping=4.0,
        ),
    },
)
