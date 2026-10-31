"""任务名映射到配置工厂、模型工厂与 Manager 类。"""
from dataclasses import dataclass
from typing import Callable
from torch import nn
from train_manager.config import TrainConfig


@dataclass(frozen=True)
class TaskSpec:
    config_factory: Callable[[], TrainConfig]
    model_factory: Callable[[], nn.Module]
    manager_cls: type


_REGISTRY: dict[str, TaskSpec] = {}


def register_task(task_id: str, spec: TaskSpec):
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError('任务名不能为空')
    if task_id in _REGISTRY:
        raise ValueError(f'任务重复注册：{task_id}')
    if not isinstance(spec, TaskSpec):
        raise TypeError('spec 必须为 TaskSpec')
    _REGISTRY[task_id] = spec


def list_tasks():
    return sorted(_REGISTRY)


def get_task(task_id: str):
    if task_id not in _REGISTRY:
        raise ValueError(f'未知任务 {task_id!r}，可用任务：{", ".join(list_tasks())}')
    return _REGISTRY[task_id]
