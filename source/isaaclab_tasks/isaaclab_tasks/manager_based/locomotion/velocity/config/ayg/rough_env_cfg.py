# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.ayg import AYG_CFG  # isort: skip


@configclass
class AygRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to ayg
        self.scene.robot = AYG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Base"
        self.events.add_base_mass.params["asset_cfg"].body_names = "Base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "Base"
        
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_Foot"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_Thigh"
        self.rewards.base_height.params["asset_cfg"].body_names = "Base"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "Base"

@configclass
class AygRoughEnvCfg_PLAY(AygRoughEnvCfg):
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
        self.events.base_external_force_torque = None
        self.events.push_robot = None
