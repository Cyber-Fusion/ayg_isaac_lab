# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.walk_these_ways.wtw_params import WalkTheseWaysParams as Params

from isaaclab_tasks.manager_based.locomotion.walk_these_ways.walk_these_ways_env_cfg import LocomotionWalkTheseWaysRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.ayg import AYG_CFG  # isort: skip


@configclass
class AygRoughWTWEnvCfg(LocomotionWalkTheseWaysRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # Switch robot to ayg and rename stuff
        self.scene.robot = AYG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        
        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        
        # rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        
        self.rewards.lin_vel_z_l2.weight = -2.0e-2
        self.rewards.ang_vel_xy_l2.weight = -2.0e-2
        self.rewards.flat_orientation_l2.weight = -1.0
        
        self.rewards.joint_vel_l2.weight = -2.0e-5
        self.rewards.joint_acc_l2.weight = -5.0e-9
        self.rewards.joint_torques_l2.weight = -2.0e-5
        
        self.rewards.base_height_l2.weight = -0.2
        self.rewards.feet_slip.weight = -8.0e-4
        
        self.rewards.action_rate_l2.weight = -2.0e-3
        self.rewards.action_smoothness_l2.weight = -2.0e-3
        
        self.rewards.feet_air_time.weight = -0.0
        self.rewards.undesired_contacts.weight = -1.0
        
        self.rewards.gait.weight = 0.5
        self.rewards.footswing_height.weight = -5.0
        self.rewards.foot_clearance.weight = 0.0
        
        self.rewards.stand_when_zero_command.weight = -0.01
        self.rewards.stand_still_when_zero_command.weight = -0.01
        
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

@configclass
class AygRoughWTWEnvCfg_PLAY(AygRoughWTWEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None
