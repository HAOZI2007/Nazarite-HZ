"""项目根目录：python -m unittest tools.test.test_collector -v"""
import unittest

import numpy as np
import torch

from algorithm.data_collect.data_collector.collector import Collector, collector
from algorithm.search.mcts.mcts import MCTSPlanner
from algorithm.networks.LeNet.model import LeNet
from scene.board import Chessboard, GameStatus


class ScriptedPlanner:
    """用固定动作序列独立验证采样标签，不依赖搜索棋力。"""
    def __init__(self, model, moves):
        self.model = model
        self.moves = moves
        self.options = []

    def search(self, board, **kwargs):
        self.options.append(kwargs)
        pi = np.zeros(board.row * board.col)
        pi[self.moves[board.step_nums]] = 1
        return pi


class CollectorTests(unittest.TestCase):
    def check_game(self, shape, moves, winner):
        board = Chessboard(*shape)
        model = object()
        planner = ScriptedPlanner(model, moves)
        trajectory = []
        c = collector(model, board, planner, trajectory, seed=42)
        samples, info = c.collect_sample()
        self.assertIs(collector, Collector)
        self.assertEqual(info['winner'], winner)
        self.assertEqual(info['total_moves'], len(moves))
        self.assertEqual(info['sample_count'], len(samples))
        self.assertEqual(info['actions'], moves)
        self.assertEqual(info['status'], GameStatus.DRAW if winner == 0 else GameStatus.WIN)
        self.assertGreaterEqual(info['elapsed_seconds'], 0)
        # 从空棋盘独立重放，验证每条样本都对应实际落子前的视角。
        replay = Chessboard(*shape)
        for (state, pi, z), action in zip(samples, moves):
            player = replay.current_player
            np.testing.assert_array_equal(state[0], replay.board == player)
            np.testing.assert_array_equal(state[1], replay.board == -player)
            self.assertEqual(state.dtype, np.float32)
            self.assertEqual(pi.dtype, np.float32)
            self.assertEqual(state.shape, (2, *shape))
            self.assertAlmostEqual(float(pi.sum()), 1)
            self.assertTrue(np.all(pi[~replay.action_mask()] == 0))
            self.assertEqual(z, 0 if winner == 0 else (1 if winner == player else -1))
            replay.step(action)
        self.assertTrue(board.terminated)
        # 连续采集自动 reset，不累计上一局 trajectory，也不修改旧样本。
        saved = [(s.copy(), p.copy(), z) for s, p, z in samples]
        again, _ = c.collect_sample()
        self.assertEqual(len(trajectory), len(moves))
        for old, copy in zip(samples, saved):
            np.testing.assert_array_equal(old[0], copy[0])
            np.testing.assert_array_equal(old[1], copy[1])
        trajectory[0][0].fill(9)
        trajectory[0][1].fill(9)
        np.testing.assert_array_equal(again[0][0], saved[0][0])
        np.testing.assert_array_equal(again[0][1], saved[0][1])

    def test_black_white_and_draw_labels(self):
        self.check_game((5, 5), [0, 5, 1, 6, 2, 7, 3, 8, 4], -1)
        self.check_game((5, 5), [20, 0, 22, 1, 24, 2, 15, 3, 17, 4], 1)
        self.check_game((2, 3), list(range(6)), 0)

    def test_invalid_policies_fail_before_bad_move(self):
        for bad in ([1, 0], [np.nan, 0, 0, 0], [-1, 2, 0, 0], [0, 0, 0, 0], [.1]*4):
            with self.subTest(bad=bad):
                model = object()
                planner = ScriptedPlanner(model, [0, 1, 2, 3])
                planner.search = lambda board, **kw: bad
                board = Chessboard(2, 2)
                c = Collector(model, board, planner)
                with self.assertRaises(ValueError):
                    c.collect_sample()
                self.assertEqual(board.step_nums, 0)
                self.assertEqual(c.trajectory, [])
        # 第一手合法，第二手却重复给已占位置概率：不得悄悄修成合法分布。
        model = object()
        board = Chessboard(2, 2)
        c = Collector(model, board, ScriptedPlanner(model, [0, 0]))
        with self.assertRaisesRegex(ValueError, '非法动作'):
            c.collect_sample()
        self.assertEqual(board.step_nums, 1)
        self.assertEqual(c.trajectory, [])

    def test_preserve_soft_policy_and_options(self):
        model = object()
        planner = ScriptedPlanner(model, [])
        def uniform(board, **kwargs):
            planner.options.append(kwargs)
            mask = board.action_mask().astype(float)
            return mask / mask.sum()
        planner.search = uniform
        c = Collector(model, Chessboard(2, 3), planner, seed=42,
                      temperature=.5, add_noise=False)
        samples, _ = c.collect_sample()
        self.assertEqual(np.count_nonzero(samples[0][1]), 6)
        np.testing.assert_allclose(samples[0][1], np.full(6, 1/6))
        self.assertTrue(all(o == {'temperature': .5, 'add_noise': False} for o in planner.options))
        # 相同采样种子、相同策略产生相同真实落子。
        d = Collector(model, Chessboard(2, 3), planner, seed=42)
        fresh = Collector(model, Chessboard(2, 3), planner, seed=42)
        self.assertEqual(d.collect_sample()[1]['actions'], fresh.collect_sample()[1]['actions'])

    def test_configuration(self):
        model = object()
        planner = ScriptedPlanner(model, [])
        for kwargs in ({'temperature': -1}, {'temperature': float('nan')},
                       {'add_noise': 'yes'}, {'trajectory': [1]}):
            with self.assertRaises(ValueError):
                Collector(model, Chessboard(2, 3), planner, **kwargs)
        with self.assertRaises(ValueError):
            Collector(object(), Chessboard(2, 3), planner)

    def test_real_network_and_mcts_integration(self):
        torch.manual_seed(42)
        model = LeNet().train()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        board = Chessboard(5, 5)
        planner = MCTSPlanner(model, simulation_limit=5, seed=42)
        c = Collector(model, board, planner, seed=42)
        samples, info = c.collect_sample()
        self.assertEqual(len(samples), board.step_nums)
        self.assertEqual(len(samples), info['sample_count'])
        self.assertTrue(board.terminated)
        self.assertTrue(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k])
        replay = Chessboard(5, 5)
        for (s, p, z), action in zip(samples, info['actions']):
            np.testing.assert_array_equal(s, replay.encoder(replay.observation()))
            self.assertTrue(np.all(p[~replay.action_mask()] == 0))
            self.assertEqual(z, 0 if info['winner'] == 0 else
                             1 if info['winner'] == replay.current_player else -1)
            replay.step(action)


if __name__ == '__main__':
    unittest.main()
