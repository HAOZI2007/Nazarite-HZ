"""均匀经验回放池，保存采集器返回的 (state, pi, z)。

buffer = ReplayBuffer(capacity=10000, seed=42)
buffer.extend(samples)
states, policies, outcomes = buffer.sample(32, device='cpu')

容量以“手数/样本条数”计，不是棋局数。超出容量淘汰最旧样本。
一个池只接收同一棋盘尺寸；不同尺寸可使用不同池。
"""

from collections import deque
from copy import deepcopy

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity=10000, *, seed=None):
        self.capacity = self._positive_integer(capacity, 'capacity')
        self._samples = deque(maxlen=self.capacity)
        self._board_shape = None
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _positive_integer(value, name):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value <= 0):
            raise ValueError(f'{name} 必须为正整数')
        return int(value)

    def __len__(self):
        return len(self._samples)

    @property
    def board_shape(self):
        """当前池的 (H,W)，未写入或 clear 后为 None。"""
        return self._board_shape

    def _validate_sample(self, sample, board_shape):
        if not isinstance(sample, (tuple, list)) or len(sample) != 3:
            raise ValueError('每条样本必须为 (state, pi, z)')
        state, pi, z = sample
        # 在 CPU 上统一保存独立的 float32 数组，不保留计算图或外部引用。
        state = np.asarray(state)
        pi = np.asarray(pi)
        if state.ndim != 3 or state.shape[0] != 2 or min(state.shape[1:]) < 1:
            raise ValueError('state 形状必须为 (2,H,W)，H、W 为正数')
        shape = state.shape[1:]
        if board_shape is not None and shape != board_shape:
            raise ValueError(f'棋盘尺寸不一致：池为 {board_shape}，样本为 {shape}')
        if (np.iscomplexobj(state)
                or not (np.issubdtype(state.dtype, np.number) or state.dtype == np.bool_)):
            raise ValueError('state 必须为数值数组')
        if not np.isin(state, (0, 1)).all() or np.any((state[0] != 0) & (state[1] != 0)):
            raise ValueError('state 两通道必须为互不重叠的 0/1 棋子编码')
        if pi.shape != (shape[0] * shape[1],):
            raise ValueError('pi 形状必须为 (H*W,)')
        if (not np.issubdtype(pi.dtype, np.number) or np.iscomplexobj(pi)
                or not np.isfinite(pi).all() or (pi < 0).any()):
            raise ValueError('pi 必须为有限非负概率数组')
        total = float(pi.sum(dtype=np.float64))
        if not np.isclose(total, 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError('pi 的概率之和必须为 1')
        occupied = np.any(state != 0, axis=0).reshape(-1)
        if np.any(pi[occupied] != 0):
            raise ValueError('pi 不能给已占用位置分配概率')
        outcome = np.asarray(z)
        if (outcome.ndim != 0 or outcome.dtype == np.bool_
                or not np.issubdtype(outcome.dtype, np.number)
                or np.iscomplexobj(outcome) or outcome.item() not in (-1, 0, 1)):
            raise ValueError('z 必须为标量 -1、0 或 +1')
        return (np.array(state, dtype=np.float32, copy=True),
                np.array(pi / total, dtype=np.float32, copy=True),
                float(outcome.item()))

    def add(self, sample):
        """添加一条 (state, pi, z)；复制数据，防止外部修改污染经验池。"""
        validated = self._validate_sample(sample, self._board_shape)
        self._samples.append(validated)
        self._board_shape = validated[0].shape[1:]

    def extend(self, samples):
        """添加一局或多局样本；全部校验通过后才写入。

        若其中有非法样本，不会部分写入，也不会提前淘汰池内旧样本。
        空输入不改变尺寸约定，超容量时仅保留最新 capacity 条。
        """
        validated = []
        shape = self._board_shape
        for sample in samples:
            item = self._validate_sample(sample, shape)
            shape = item[0].shape[1:]
            validated.append(item)
        self._samples.extend(validated)
        self._board_shape = shape

    def sample(self, batch_size, *, device='cpu'):
        """均匀无放回抽样，返回同设备的三个 float32 Tensor。

        states: (B,2,H,W)，policies: (B,H*W)，outcomes: (B,)。
        batch_size 不得超过当前样本数；抽样不会移除数据。
        返回数据与池内数组不共享内存，调用方可安全修改。
        """
        batch_size = self._positive_integer(batch_size, 'batch_size')
        if len(self) < batch_size:
            raise ValueError(f'样本不足：当前 {len(self)} 条，要求 {batch_size} 条')
        indices = self.rng.choice(len(self), size=batch_size, replace=False)
        # deque 随机索引成本较高，先建立引用快照，再按随机下标访问。
        snapshot = list(self._samples)
        states, policies, outcomes = zip(*(snapshot[int(i)] for i in indices))
        return (
            torch.from_numpy(np.stack(states)).to(device=device),
            torch.from_numpy(np.stack(policies)).to(device=device),
            torch.tensor(outcomes, dtype=torch.float32, device=device),
        )

    def clear(self):
        """清空样本与棋盘尺寸约定；不重置随机数生成器。"""
        self._samples.clear()
        self._board_shape = None

    def state_dict(self):
        """独立的 CPU 快照，包含抽样随机数状态。"""
        return {'capacity': self.capacity,
                'samples': [(s.copy(), p.copy(), z) for s, p, z in self._samples],
                'rng_state': deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        """先验证临时池，失败时不改变当前池。"""
        if state['capacity'] != self.capacity:
            raise ValueError('断点经验池容量与配置不一致')
        if len(state['samples']) > self.capacity:
            raise ValueError('断点样本数超过经验池容量')
        temporary = ReplayBuffer(self.capacity)
        for sample in state['samples']:
            validated = temporary._validate_sample(sample, temporary._board_shape)
            # 快照已经归一化；再次归一化可能改变浮点尾数，影响续训复现。
            if sample[0].dtype != np.float32 or sample[1].dtype != np.float32:
                raise ValueError('断点样本必须为 float32 数组')
            temporary._samples.append((validated[0], sample[1].copy(), validated[2]))
            temporary._board_shape = validated[0].shape[1:]
        temporary.rng.bit_generator.state = deepcopy(state['rng_state'])
        self._samples = temporary._samples
        self._board_shape = temporary._board_shape
        self.rng = temporary.rng
