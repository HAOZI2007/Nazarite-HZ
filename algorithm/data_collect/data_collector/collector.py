"""串行 MCTS 自博弈采集，不更新网络，也不直接写入经验池。

collector = Collector(model, env, planner, seed=42)
samples, game_info = collector.collect_sample()
# samples 中每项为 (落子前 state, 搜索策略 pi, 终局结果 z)。
# replay_buffer.extend(samples) 由调用方负责。
"""

import math
import time

import numpy as np


class Collector:
    def __init__(self, model, board, planner, trajectory=None, *, seed=None,
                 temperature=1.0, add_noise=True):
        """复用同一个模型和搜索器；每次 collect_sample 从空棋盘开始。

        trajectory 可传入空列表，供查看最近一次对局的 (state, pi, player)。
        每局开始清空它，因此不要将经验池作为 trajectory 传入。
        temperature 控制整局的动作采样分布，0 为贪心，1 按访问次数采样。
        seed 控制真实动作采样；MCTS 内部随机数由 planner 的 seed 控制。
        """
        if planner.model is not model:
            raise ValueError("采集器与 planner 必须使用同一个 model 对象")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature 必须为有限非负数")
        if not isinstance(add_noise, (bool, np.bool_)):
            raise ValueError("add_noise 必须为布尔值")
        if trajectory is not None and (not isinstance(trajectory, list) or trajectory):
            raise ValueError("trajectory 必须为 None 或空列表，不能混入其他对局记录")
        self.train_model = model
        self.env = board
        self.mcts_planner = planner
        self.trajectory = [] if trajectory is None else trajectory
        self.temperature = float(temperature)
        self.add_noise = bool(add_noise)
        self.rng = np.random.default_rng(seed)

    def _validated_policy(self, policy):
        """拒绝错误分布，不静默将非法落子改为随机合法动作。"""
        policy = np.asarray(policy, dtype=np.float64)
        expected_shape = (self.env.row * self.env.col,)
        if policy.shape != expected_shape:
            raise ValueError(f"搜索策略形状应为 {expected_shape}，实际为 {policy.shape}")
        if not np.isfinite(policy).all() or (policy < 0).any():
            raise ValueError("搜索策略必须为有限非负数")
        if not np.isclose(policy.sum(), 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError("搜索策略概率之和必须为 1")
        if np.any(policy[~self.env.action_mask()] != 0):
            raise ValueError("搜索策略给非法动作分配了非零概率")
        # 消除浮点归一化误差，满足 Generator.choice 的精度要求。
        policy = policy.copy()
        policy /= policy.sum()
        return policy

    def collect_sample(self):
        """完整采集一局，返回 (samples, game_info)。

        samples: [(state, pi, z), ...]，一手真实落子对应一条。
          state: float32 ndarray，(2,H,W)，当前方/对手视角。
          pi: float32 ndarray，(H*W,)，完整搜索概率分布。
          z: Python float，当前样本行动方最终赢/平/输对应 +1/0/-1。
        game_info: winner、status、total_moves、sample_count、actions、elapsed_seconds。

        环境会 reset 并最终停留在终局。模型通过 MCTS 仅推理，不做梯度更新。
        中途异常会清空未完成记录并抛出，不把不完整对局当成平局返回。
        """
        self.trajectory.clear()
        started = time.perf_counter()
        actions = []
        try:
            obs = self.env.reset()
            # 最多填满棋盘，避免环境终局逻辑出错造成无限循环。
            for _ in range(self.env.row * self.env.col):
                player = obs['current_player']
                state = self.env.encoder(obs)
                pi = self.mcts_planner.search(
                    self.env,
                    temperature=self.temperature,
                    add_noise=self.add_noise,
                )
                pi = self._validated_policy(pi)
                action = int(self.rng.choice(len(pi), p=pi))
                # 记录落子前局面和完整 pi，不将采样动作转换为 one-hot 目标。
                self.trajectory.append((state.copy(), pi.astype(np.float32), player))
                obs, terminated, info = self.env.step(action)
                actions.append(action)
                if terminated:
                    break
            if not obs['terminated']:
                raise RuntimeError("棋盘填满后环境仍未终局，拒绝返回未完成样本")
            winner = obs['winner']
            if winner not in (self.env.b, self.env.w, 0):
                raise ValueError(f"非法赢家标记：{winner!r}")
            samples = []
            for state, pi, player in self.trajectory:
                z = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
                # 与 trajectory 分离，后续清空或修改记录不影响已返回样本。
                samples.append((state.copy(), pi.copy(), z))
            game_info = {
                'winner': winner,
                'status': obs['status'],
                'total_moves': len(actions),
                'sample_count': len(samples),
                'actions': actions,
                'elapsed_seconds': time.perf_counter() - started,
            }
            return samples, game_info
        except Exception:
            self.trajectory.clear()
            raise


# 兼容原先的 from ...collector import collector。
collector = Collector
