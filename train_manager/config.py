"""任务默认配置；命令行可覆盖常用字段。"""
from dataclasses import dataclass
import math


@dataclass
class TrainConfig:
    rows: int = 8
    cols: int = 8
    seed: int = 42
    device: str = 'auto'
    simulations: int = 100
    temperature: float = 1.0
    add_noise: bool = True
    capacity: int = 10000
    min_samples: int = 128
    batch_size: int = 32
    iterations: int = 100  # 目标总轮数，续训时不是额外增加的轮数
    games_per_iteration: int = 5
    updates_per_iteration: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    output_dir: str = 'logs'
    save_interval: int = 10

    def validate(self):
        for name in ('rows', 'cols', 'simulations', 'capacity', 'min_samples',
                     'batch_size', 'iterations', 'games_per_iteration',
                     'updates_per_iteration', 'save_interval'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} 必须为正整数')
        if not self.capacity >= self.min_samples >= self.batch_size:
            raise ValueError('必须满足 capacity >= min_samples >= batch_size')
        for name in ('learning_rate', 'weight_decay', 'value_weight', 'temperature'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} 必须为有限非负数')
        if self.learning_rate == 0:
            raise ValueError('learning_rate 必须大于零')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('seed 必须为非负整数')
        if not isinstance(self.add_noise, bool):
            raise ValueError('add_noise 必须为布尔值')
        if not isinstance(self.device, str) or not self.device:
            raise ValueError('device 必须为有效设备字符串或 auto')
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError('output_dir 不能为空')
