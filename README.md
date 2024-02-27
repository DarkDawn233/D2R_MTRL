# Code Instruction for D2R

This instruction hosts the PyTorch implementation of "**Not All Tasks Are Equally Difficult: Multi-Task Deep Reinforcement Learning with Dynamic Depth Routing**" (AAAI 2024, [Link](https://arxiv.org/abs/2312.14472)) with the [Meta-World](https://github.com/Farama-Foundation/Metaworld) benchmark.

**NOTE**: Since [MetaWorld](https://meta-world.github.io) is under active development,  we perform all the experiments on the following commit-id: https://github.com/Farama-Foundation/Metaworld/commit/04be337a12305e393c0caf0cbf5ec7755c7c8feb



## Setup

1. Set up the working environment: 

Required packages: pytorch==1.13.1, json5, tensorboardX, posix_ipc, scipy

```shell
pip install -r requirements.txt
```

2. Set up the Meta-World benchmark: 

Please follow the [instructions](https://github.com/openai/mujoco-py#install-mujoco) to install the mujoco-py package first.

```shell
pip install git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb
```



## Training

To train `D2R` on the `MT10-Rand` setting, 

```shell
python train.py --config config/mt10/depthroute_rand.json --id mt10_rand --seed 1
```

Change the `config` accordingly for other setting (e.g. `MT50-Rand`). 

All results will be saved in the `log` folder. 

The config file `config/mt10/depthroute_rand.json` contains default hyperparameters for D2R.



## See Also

See [Meta-World](https://github.com/Farama-Foundation/Metaworld), [mujoco-py](https://github.com/openai/mujoco-py) for additional instructions.