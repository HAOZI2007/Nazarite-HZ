"""Trainer 的真实闭环验证（默认 CPU，无需保存模型）。

项目根目录运行：
    python -m tools.test.test_trainer
或加入回归测试：
    python -m unittest tools.test.test_trainer -v

使用真实棋盘、LeNet、MCTS、Collector、ReplayBuffer 和 PolicyValueTrainer，
验证两轮“采集 → 入池 → 参数更新”；这不用于判断棋力是否提升。
"""

import math
import unittest

import numpy as np
import torch

from algorithm.data_collect.data_collector.collector import Collector
from algorithm.data_collect.sample_pool.pool import ReplayBuffer
from algorithm.networks.LeNet.model import LeNet
from algorithm.search.mcts.mcts import MCTSPlanner
from algorithm.training.trainer import PolicyValueTrainer
from scene.board import Chessboard


def require(condition, message):
    """不用 assert，保证 python -O 下仍然执行关键检查。"""
    if not condition:
        raise AssertionError(message)


def snapshot(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def verify_game(samples, info, board_size):
    """独立重放动作，核对每一条样本的时刻、玩家视角、策略与 z。"""
    board = Chessboard(board_size, board_size)
    require(len(samples) == len(info['actions']) == info['total_moves'], '样本数和真实落子数不一致')
    for (state, pi, z), action in zip(samples, info['actions']):
        np.testing.assert_array_equal(state, board.encoder(board.observation()))
        require(state.dtype == np.float32 and pi.dtype == np.float32, '样本类型不是 float32')
        require(np.isfinite(pi).all() and (pi >= 0).all(), '策略含异常概率')
        require(np.isclose(pi.sum(), 1), '策略未归一化')
        require(np.all(pi[~board.action_mask()] == 0), '非法落点概率非零')
        require(action in board.legal_actions(), '真实动作非法')
        expected_z = 0.0 if info['winner'] == 0 else (
            1.0 if info['winner'] == board.current_player else -1.0
        )
        require(z == expected_z, '终局标签的玩家视角错误')
        board.step(action)
    require(board.terminated and board.winner == info['winner'], '重放终局与采集结果不一致')


def run_closed_loop(batch_size=8, *, verbose=False):
    """运行两轮，第二轮复用第一轮更新后的同一网络和同一优化器。"""
    torch.manual_seed(42)
    device = torch.device('cpu')
    model = LeNet().to(device)
    env = Chessboard(5, 5)
    planner = MCTSPlanner(model, simulation_limit=5, seed=42)
    collector = Collector(model, env, planner, seed=42)
    buffer = ReplayBuffer(capacity=100, seed=42)
    trainer = PolicyValueTrainer(model)
    require(planner.model is collector.train_model is trainer.model is model,
            '搜索、采集和训练没有共享同一个模型')

    reports = []
    for iteration in range(1, 3):
        before_search = snapshot(model)
        previous_grads = {
            name: None if p.grad is None else p.grad.detach().clone()
            for name, p in model.named_parameters()
        }
        modes = [module.training for module in model.modules()]
        samples, info = collector.collect_sample()
        verify_game(samples, info, 5)
        # 搜索不会偷偷更新参数、覆盖训练留下的梯度或改变模块模式。
        for name, tensor in model.state_dict().items():
            require(torch.equal(tensor, before_search[name]), f'采集修改了参数 {name}')
        for name, parameter in model.named_parameters():
            old_grad = previous_grads[name]
            require(parameter.grad is None if old_grad is None else
                    parameter.grad is not None and torch.equal(parameter.grad, old_grad),
                    f'采集修改了梯度 {name}')
        require(modes == [m.training for m in model.modules()], '搜索未恢复模型模式')

        old_length = len(buffer)
        buffer.extend(samples)
        require(len(buffer) == min(buffer.capacity, old_length + len(samples)), '经验池计数错误')
        require(len(buffer) >= batch_size, '测试局样本不足 batch_size')
        batch = buffer.sample(batch_size, device=device)
        expected = [(batch_size, 2, 5, 5), (batch_size, 25), (batch_size,)]
        for tensor, shape in zip(batch, expected):
            require(tuple(tensor.shape) == shape, '训练批次维度错误')
            require(tensor.dtype == torch.float32 and tensor.device == device, '数据类型或设备错误')
            require(not tensor.requires_grad, '样本不应要求梯度')
        torch.testing.assert_close(batch[1].sum(1), torch.ones(batch_size))

        before_update = snapshot(model)
        metrics = trainer.update(batch)  # 调用实际 Trainer，不在测试中重写训练逻辑。
        require(all(math.isfinite(v) for v in metrics.values()), '损失不是有限数')
        require(math.isclose(metrics['loss'], metrics['policy_loss'] +
                             trainer.value_weight * metrics['value_loss'], rel_tol=1e-5),
                '总损失与分项不一致')
        for name, parameter in model.named_parameters():
            require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(),
                    f'参数缺少有效梯度：{name}')
        # 共享层、策略头、价值头都必须有实际参数更新。
        for group in ('c1.', 'c2.', 'policy_head.', 'value_head.'):
            require(any(not torch.equal(before_update[name], parameter.detach())
                        for name, parameter in model.named_parameters() if name.startswith(group)),
                    f'模块参数未更新：{group}')
        require(all(torch.isfinite(p).all().item() for p in model.parameters()), '更新后参数非有限数')
        # Adam 的状态连续推进，不能每次更新重建优化器。
        for parameter in model.parameters():
            require(trainer.optimizer.state[parameter]['step'].item() == iteration,
                    '优化器状态未连续推进')
        reports.append({'iteration': iteration, 'moves': info['total_moves'],
                        'winner': info['winner'], 'buffer_size': len(buffer), **metrics})
        if verbose:
            print(f"第 {iteration} 轮 | {info['total_moves']} 手 | winner={info['winner']} | "
                  f"池内 {len(buffer)} 条 | batch={batch_size} | "
                  f"loss={metrics['loss']:.6f} | "
                  f"policy={metrics['policy_loss']:.6f} | value={metrics['value_loss']:.6f}")
    return reports


class TrainerClosedLoopTests(unittest.TestCase):
    def test_real_closed_loop(self):
        reports = run_closed_loop(batch_size=8)
        self.assertEqual(len(reports), 2)
        self.assertGreater(reports[1]['buffer_size'], reports[0]['buffer_size'])

    def test_batch_size_one(self):
        # 特别检查 squeeze(-1) 在 B=1 时仍保留批次维度。
        self.assertEqual(len(run_closed_loop(batch_size=1)), 2)


if __name__ == '__main__':
    run_closed_loop(verbose=True)
    run_closed_loop(batch_size=1)
    print('通过：两轮真实自博弈与 Trainer 更新；样本视角、梯度、双头参数、优化器状态及 B=1 均正常。')
