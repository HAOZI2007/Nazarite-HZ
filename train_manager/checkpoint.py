"""在轮次边界保存训练快照；只加载自己生成、可信的断点。"""
import os
from pathlib import Path
import tempfile

import torch


FORMAT_VERSION = 1


def save_checkpoint(path, payload):
    """同目录临时文件写入成功后替换，失败时保留原文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save({'format_version': FORMAT_VERSION, **payload}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path):
    # 包含 NumPy 经验池和 RNG 状态，不是仅含张量的权重文件。
    data = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(data, dict) or data.get('format_version') != FORMAT_VERSION:
        raise ValueError('不支持的 checkpoint 格式或版本')
    required = {'task_id', 'model_class', 'config', 'model', 'optimizer',
                'counters', 'buffer', 'rng'}
    if not required <= data.keys():
        raise ValueError(f'checkpoint 缺少字段：{sorted(required - data.keys())}')
    return data
