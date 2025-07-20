# Isaac Lab Reinforcement Learning Guide

## Online Guides

The introduction to Isaac Lab [here](https://isaac-sim.github.io/IsaacLab/main/index.html) is very good.

All the pages in the [Getting Started](https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html#) section are useful.
- The [Walkthrough](https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/index.html) gives a very general and high-level overview of the framework and of Reinforcement Learning (RL). They can be quickly skimmed.
- The [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html) provide a more in-depth look at how to perform certain tasks and simple examples.
  For RL, take a look at: 
  - [Adding a new robot](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/add_new_robot.html): skim
  - [Manager-Based Env](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_base_env.html)
  - [Manager-Based RL Env](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html)
  - [Direct Workflow RL Env](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html)
  - [Registering an Env](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/register_rl_env_gym.html)
  - [Training with an RL Agent](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_rl_training.html)
  - [Modifying an existing Direct RL Env](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/modify_direct_rl_env.html)
- [How to Guides](https://isaac-sim.github.io/IsaacLab/main/source/how-to/index.html) give a deeper look compared to the tutorials. For RL, take a look at:
  - [Curriculum utilities](https://isaac-sim.github.io/IsaacLab/main/source/how-to/curriculums.html)

## Codebase

Ayg robot description is in [`source/isaaclab_assets/isaaclab_assets/robots/ayg.py`](source/isaaclab_assets/isaaclab_assets/robots/ayg.py).
The USD is in `source/isaaclab_assets/data/Robots/Ayg/`.

The RL environments are in `source/isaaclab_tasks/isaaclab_tasks`.
They are divided into:
- `direct`: single-file implementation, slightly more efficient,
- `manager_based`: modular implementation, more difficult to understand and modify, more reusable.

The policies are currently correctly implemented as `manager_based`, while the `direct` implementation is not yet complete.
Three `manager_based` environments are available for Ayg:
- Classic policy in `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/rough_env.py` for rough terrains and flat_env for flat terrains. Simplest to train, but not very good performance.
- Spot-like env in `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot_inspired_env_cfg.py`. Inspired by the policy available for Spot in Isaac Lab. Good performance. Does stop with null velocity commands. Slow to train.
- Walk-these-ways-inspired policy in `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/walk_these_ways/walk_these_ways_env_cfg.py`. Inspired by the Walk These Ways paper. Good performance, good training speed, can control the gait.

In the `manager_based` implementation, there is a general file for the environment configuration. It can use classes and functions for the curriculum, observations, rewards, terminations, and others part of the pipeline, usually in the `mdp` folder.
The `config` folder collects the specific configurations for the various robots, which specify the gains, the joint names, and other robot-specific parameters.

Both the `direct` and `manager_based` implementations build upon the code implemented in `source/isaaclab/isaaclab`, which contains, among others, specifications on the envs, managers (for commands, observations, and rewards), and other stuff.
Seeing their implementation is sometimes useful to understand what is happening.

Both in the `direct` and `manager_based` implementations, new RL environments are registered through the `__init__.py` file in the `config/<robot_name>` folder.