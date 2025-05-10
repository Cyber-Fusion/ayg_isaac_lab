(2025-05-10)
---
- Modified [`source/isaaclab/managers/reward_manager.py`](./source/isaaclab/isaaclab/managers/reward_manager.py) to enable three types of rewards:
    - `classic`: The classic reward function: $R = \sum_i^n r_i$
    - `always_positive`: Every negative reward term is clipped to zero.
    - `exp_negative`: The reward is: $r_\text{positive} \cdot \operatorname{exp}(r_\text{neg scale} \cdot r_\text{negative})$.

  These reward types can be used by setting `reward_type` and `negative_reward_scale` in the `reward_manager` config.
  
  `exp_negative` is inspired by [Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior](https://doi.org/10.48550/arXiv.2212.03238) and used here in the `walk_these_ways` task.