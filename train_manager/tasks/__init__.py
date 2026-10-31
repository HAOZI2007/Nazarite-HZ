"""导入本模块注册内置任务，不创建网络或启动训练。"""
from algorithm.networks.LeNet.model import LeNet
from train_manager.manager import TrainManager
from train_manager.registry import TaskSpec, register_task
from .gomoku import (gomoku_lenet_8x8_config, gomoku_lenet_15x15_config,
                     gomoku_lenet_smoke_config)

for name, factory in (
    ('Gomoku-LeNet-8x8', gomoku_lenet_8x8_config),
    ('Gomoku-LeNet-15x15', gomoku_lenet_15x15_config),
    ('Gomoku-LeNet-Smoke', gomoku_lenet_smoke_config),
):
    register_task(name, TaskSpec(factory, LeNet, TrainManager))
