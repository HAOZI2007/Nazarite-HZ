"""项目根目录运行：python -m unittest tools.test.test_pool -v"""
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from algorithm.data_collect.sample_pool.pool import ReplayBuffer
from algorithm.data_collect.data_collector.collector import Collector
from algorithm.search.mcts.mcts import MCTSPlanner
from algorithm.networks.LeNet.model import LeNet
from scene.board import Chessboard


def sample(action=0, shape=(2, 3), z=0):
    state = np.zeros((2, *shape), dtype=np.float32)
    pi = np.zeros(shape[0] * shape[1], dtype=np.float32)
    pi[action] = 1
    return state, pi, z


class ReplayBufferTests(unittest.TestCase):
    def test_capacity_fifo_and_no_replacement(self):
        buffer = ReplayBuffer(3, seed=42)
        buffer.extend(sample(i) for i in range(5))
        self.assertEqual(len(buffer), 3)
        self.assertEqual(buffer.board_shape, (2, 3))
        _, policies, _ = buffer.sample(3)
        self.assertEqual(set(policies.argmax(1).tolist()), {2, 3, 4})
        self.assertEqual(len(buffer), 3)
        buffer.add(sample(5))
        self.assertEqual(set(buffer.sample(3)[1].argmax(1).tolist()), {3, 4, 5})

    def test_copy_isolation_and_types(self):
        buffer = ReplayBuffer(4)
        s, p, z = sample(z=1)
        buffer.add((s, p, z))
        s.fill(9)
        p.fill(9)
        states, policies, outcomes = buffer.sample(1)
        self.assertEqual(tuple(states.shape), (1, 2, 2, 3))
        self.assertEqual(tuple(policies.shape), (1, 6))
        self.assertEqual(tuple(outcomes.shape), (1,))
        for t in (states, policies, outcomes):
            self.assertEqual(t.dtype, torch.float32)
            self.assertEqual(t.device.type, 'cpu')
            self.assertFalse(t.requires_grad)
        self.assertEqual(states.sum().item(), 0)
        self.assertEqual(policies[0, 0].item(), 1)
        self.assertEqual(outcomes.item(), 1)
        states.fill_(9)
        policies.fill_(9)
        outcomes.fill_(9)
        again = buffer.sample(1)
        self.assertEqual(again[0].sum().item(), 0)
        self.assertEqual(again[1].sum().item(), 1)
        self.assertEqual(again[2].item(), 1)

    def test_validation_and_atomic_extend(self):
        buffer = ReplayBuffer(2)
        buffer.add(sample(0))
        invalid = []
        s, p, z = sample()
        invalid += [(s[0], p, z), (s, p.reshape(2, 3), z), (s, p, .5),
                    (s, p, True), (s, p, [1]), (s, p, float('nan')),
                    sample(shape=(3, 3)), (s, np.full(6, .1), z)]
        for kind in ('negative', 'nan', 'occupied', 'overlap', 'complex'):
            s, p, z = sample()
            if kind == 'negative': p[1] = -.1
            if kind == 'nan': p[1] = np.nan
            if kind == 'occupied': s[0, 0, 0] = 1
            if kind == 'overlap':
                s = s.astype(bool)
                s[:, 0, 1] = True
            if kind == 'complex': s = s.astype(complex)
            invalid.append((s, p, z))
        for bad in invalid:
            with self.subTest(bad=str(bad)):
                with self.assertRaises(ValueError):
                    buffer.extend([sample(1), bad])
                self.assertEqual(len(buffer), 1)
                self.assertEqual(buffer.sample(1)[1].argmax().item(), 0)
        empty = ReplayBuffer()
        with self.assertRaises(ValueError):
            empty.extend([sample(), sample(shape=(3, 4))])
        self.assertEqual(len(empty), 0)
        self.assertIsNone(empty.board_shape)

    def test_clear_seed_and_insufficient_samples(self):
        a, b = ReplayBuffer(seed=42), ReplayBuffer(seed=42)
        a.extend([sample(i) for i in range(6)])
        b.extend([sample(i) for i in range(6)])
        torch.testing.assert_close(a.sample(3)[1], b.sample(3)[1])
        for size in (0, -1, True, 1.5, 7):
            with self.assertRaises(ValueError): a.sample(size)
        for cap in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): ReplayBuffer(cap)
        a.clear()
        self.assertEqual(len(a), 0)
        self.assertIsNone(a.board_shape)
        a.extend([])
        self.assertIsNone(a.board_shape)
        with self.assertRaises(ValueError): a.sample(1)
        a.add(sample(shape=(3, 4)))
        self.assertEqual(a.board_shape, (3, 4))

    def test_collector_to_buffer_to_optimizer(self):
        torch.manual_seed(42)
        model = LeNet()
        board = Chessboard(5, 5)
        planner = MCTSPlanner(model, simulation_limit=5, seed=42)
        samples, info = Collector(model, board, planner, seed=42).collect_sample()
        buffer = ReplayBuffer(100, seed=42)
        buffer.extend(samples)
        self.assertEqual(len(buffer), info['total_moves'])
        states, pi, z = buffer.sample(4)
        torch.testing.assert_close(pi.sum(1), torch.ones(4))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = [p.detach().clone() for p in model.parameters()]
        model.train()
        optimizer.zero_grad()
        logits, values = model(states)
        loss = -(pi * F.log_softmax(logits, 1)).sum(1).mean() + F.mse_loss(values.squeeze(-1), z)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))
        # 同一个 planner 能继续使用已更新的网络完成下一局。
        more, _ = Collector(model, board, planner, seed=43).collect_sample()
        buffer.extend(more)
        self.assertEqual(len(buffer), len(samples) + len(more))


if __name__ == '__main__':
    unittest.main()
