# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import AygRoughWTWEnvCfg


@configclass
class AygFlatWTWEnvCfg(AygRoughWTWEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # override rewards
        # self.rewards.track_lin_vel_xy_exp.weight = 3.0
        # self.rewards.track_ang_vel_z_exp.weight = 1.5

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class AygFlatWTWEnvCfg_PLAY(AygFlatWTWEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None
