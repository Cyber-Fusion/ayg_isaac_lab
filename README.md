# Ayg Isaac Lab

Ayg Isaac Lab is a fork of [Isaac Lab](https://isaac-sim.github.io/IsaacLab) that is tailored for training the Ayg robot.

## Table of Contents

- [Ayg Isaac Lab](#ayg-isaac-lab)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Docker](#docker)
  - [Usage](#usage)
  - [Troubleshooting](#troubleshooting)

## Installation

### Docker

The "full" installation steps provided by NVIDIA are available [here](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html).
Alternatively, follow the steps below.

---

Install [Docker Community Edition](https://docs.docker.com/engine/install/ubuntu/) (ex Docker Engine).
You can follow the installation method through `apt`.
Note that it makes you verify the installation by running `sudo docker run hello-world`.
It is better to avoid running this command with `sudo` and instead follow the post installation steps first and then run the command without `sudo`.

Follow with the [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) for Linux.

Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) (nvidia-docker2).

Join the [NVIDIA Developer Program](https://developer.nvidia.com/login).

Install NGC CLI following the steps detailed [here](https://org.ngc.nvidia.com/setup/installers/cli).

Generate an [NGC API key](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#generating-api-key).
Remember to save the API key as it will not be shown again.

Log in to NGC with
```shell
ngc config set
```

Log in to the NVIDIA Container Registry with
```shell
docker login nvcr.io
```

For the username, enter `$oauthtoken` exactly as shown. It is a special username that is used to authenticate with NGC.
For the password, enter the NGC API key you generated.
```
Username: $oauthtoken
Password: <Your NGC API Key>
```

---

Build the image and bring up the container in detached mode with
```shell
./docker/container.py start
```
The image will be rebuilt every time the files are changed.

Begin a new bash process in an existing Isaac Lab container with
```shell
./docker/container.py enter base
```

Bring down the container and remove it with
```shell
./docker/container.py stop
```

## Usage

Train the robot with
```shell
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Ayg-v0 --headless
```

Visualize the training performance with
```shell
./isaaclab.sh -p -m tensorboard.main --logdir logs/rl_games/ayg_flat_direct/path_to_log
```

Test the trained robot with
```shell
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Ayg-Direct-v0 --num_envs 32 --checkpoint /path/to/checkpoint
```

## Troubleshooting

- **No GPU available in Docker**: running `nvidia-smi` in the Docker container returns `Failed to initialize NVML: Unknown Error`.\
  Solution:
  - Run `sudo nano /etc/nvidia-container-runtime/config.toml`, set `no-cgroups = false`, and save (Ctrl + X and then Y).
  - Restart Docker with `sudo systemctl restart docker`.
