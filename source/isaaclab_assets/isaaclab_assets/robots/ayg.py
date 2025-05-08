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
    effort_limit=60.0,
    saturation_effort=40.0,
    velocity_limit=20.0,
    stiffness=40.0,
    damping=1.0,
    friction=0.0,
)
"""Configuration for AYG's motor with DC actuator model."""


##
# Configuration - Articulation.
##

AYG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Ayg/ayg/ayg.usd",
        usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Robots/ayg/ayg.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
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
