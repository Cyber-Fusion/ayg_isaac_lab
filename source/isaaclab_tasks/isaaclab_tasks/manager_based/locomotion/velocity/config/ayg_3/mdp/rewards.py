from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
    

# =============================== Task Rewards ============================== #

class GaitReward(ManagerTermBase):
    """
    Reward term for the gait of the robot.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        
        # ============================ Parameters =========================== #
        
        self.step_frequency: float = cfg.params["step_frequency"]
        self.phi = cfg.params["phi"]
        if len(self.phi) != 3:
            raise ValueError("The phi parameter must be a list of three floats between 0 and 1.")
        if not all(isinstance(i, (int, float)) for i in self.phi):
            raise ValueError("The phi parameter must be a list of three floats between 0 and 1.")
        if not all(0 <= i <= 1 for i in self.phi):
            raise ValueError("The phi parameter must be a list of three floats between 0 and 1.")
        
        self.sigma: float = cfg.params["sigma"]
        
        self.sigma_cf = cfg.params["sigma_cf"]
        
        self.sigma_cv = cfg.params["sigma_cv"]
        
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensor[cfg.params["sensor_cfg"].name]
        
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        
        self.footstep_height = cfg.params["footstep_height"]
        
        # ======================== Internal Variables ======================= #
        
        self.c_feet = torch.zeros(1, 4, device=self.env.device)
        
    def _cdf(self, x, sigma):
        """
        Cumulative density function of a normal distribution.
        """
        
        return 1 / (sigma * torch.sqrt(2 * torch.pi)) * torch.exp(
            -0.5 * ((x - 0) / sigma) ** 2
        )

    def _compute_c_feet(self):
        t = self.env.episode_length_buf * self.env.step_dt
        
        t_lf = torch.clip(self.step_frequency * t + self.phi[0] + self.phi[2], min=0, max=1)
        t_rf = torch.clip(self.step_frequency * t + self.phi[1] + self.phi[2], min=0, max=1)
        t_lh = torch.clip(self.step_frequency * t + self.phi[1], min=0, max=1)
        t_rh = torch.clip(self.step_frequency * t + self.phi[0], min=0, max=1)
        
        t_feet = [t_lf, t_rf, t_lh, t_rh]
        
        for i, t_foot in enumerate(t_feet):
            
            self.c_feet[0, i] = self._cdf(t_foot, self.sigma) \
                    * (1 - self._cdf(t_foot - 0.5, self.sigma)) \
                + self._cdf(t_foot - 1, self.sigma) \
                    * (1 - self._cdf(t_foot - 1.5, self.sigma))
        
    def swing_phase_tracking_force(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg
    ):
        contact_sensor: ContactSensor = env.scene.sensor[self.contact_sensor.name]
        
        net_contact_forces = contact_sensor.data.net_forces_w
        
        return torch.sum((1 - self.c_feet) * torch.exp(
            - torch.norm(net_contact_forces[:, sensor_cfg.params["sensor_cfg"].body_ids], dim=-1) \
                / self.sigma_cf
        ), dim=-1)
        
    def stance_phase_tracking_vel(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
    ):
        asset: RigidObject = env.scene[asset_cfg.name]
        
        foot_planar_velocity = torch.linalg.norm(
            asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
        
        return (1 - self.c_feet) * torch.exp(
            - foot_planar_velocity / self.sigma_cv
        )
        
    def footswing_height_tracking(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
    ):
        asset: RigidObject = env.scene[asset_cfg.name]
        
        foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
        
        return torch.sum(
            (foot_height - self.footstep_height) ** 2 * self.c_feet, dim=-1
        )
        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        step_frequency: float,
        footstep_height: float,
        synced_feet_pair_names: tuple,
        phi: tuple,
        sigma: float,
        sigma_cf: float,
        sigma_cv: float,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ):
        self._compute_c_feet()
        
        return self.swing_phase_tracking_force(env, sensor_cfg) \
            + self.stance_phase_tracking_vel(env, asset_cfg) \
            + self.footswing_height_tracking(env, asset_cfg)
