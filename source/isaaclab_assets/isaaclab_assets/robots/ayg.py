"""Configuration for the Ayg robots.

The following configuration parameters are available:

* :obj:`AYG_CFG`: The AYG robot

Reference:

* https://github.com/AYG/ayg_description

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration - Actuators.
##

AYG_MOTOR_SIMPLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    saturation_effort=60.0,
    effort_limit=30.0,
    velocity_limit=10.0,
    stiffness=40.0,
    damping=0.5,
    friction=0.0,
)
"""Configuration for AYG's motor with DC actuator model."""


##
# Configuration - Articulation.
##

AYG_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        asset_path="/workspace/isaaclab/source/isaaclab_assets/data/Robots/ayg_description/urdf/ayg.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            ".*HAA": 0.0,       # all HAA
            ".*HFE": 0.0,       # all HFE
            ".*KFE": 0.0,       # all KFE
        },
        joint_vel={".*": 0.0},
    ),
    actuators={"legs": AYG_MOTOR_SIMPLE_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of Ayg robot using DC actuator."""
