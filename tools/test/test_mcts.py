"""项目根目录运行：python -m unittest tools.test.test_mcts -v"""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
from torch import nn
from algorithm.search.mcts.mcts import MCTSNode, MCTSPlanner
from scene.board import Chessboard


class UniformNet(nn.Module):
    """零价值、均匀先验，验证搜索逻辑而非模型棋力。"""
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        assert not self.training and not torch.is_grad_enabled()
        return x.new_zeros((len(x), x.shape[-2] * x.shape[-1])), x.new_zeros((len(x), 1))


class MCTSTests(unittest.TestCase):
    def test_backup_and_parent_perspective(self):
        root = MCTSNode(-1)
        root.expand([0, 1], [.5, .5])
        child = root.children[0]
        child.expand([2], [1.0])
        leaf = child.children[2]
        leaf.backup(.6)
        self.assertAlmostEqual(root.q_value, .6)
        self.assertAlmostEqual(child.q_value, -.6)
        self.assertAlmostEqual(leaf.q_value, .6)
        self.assertIs(root.select_child(.01), child)

    def test_policy_and_state_isolation(self):
        board = Chessboard(5, 7)
        board.step(3)
        snapshot = board.observation()
        model = UniformNet().train()
        planner = MCTSPlanner(model, simulation_limit=40, seed=42)
        pi = planner.search(board)
        self.assertEqual(pi.shape, (35,))
        self.assertAlmostEqual(pi.sum(), 1)
        self.assertEqual(pi[3], 0)
        self.assertTrue(np.isfinite(pi).all() and (pi >= 0).all())
        self.assertEqual(planner.root.visit_count, 40)
        self.assertEqual(sum(c.visit_count for c in planner.root.children.values()), 40)
        np.testing.assert_array_equal(board.board, snapshot['board'])
        self.assertEqual(board.step_nums, 1)
        self.assertEqual(board.current_player, 1)
        self.assertFalse(board.terminated)
        self.assertEqual(board.status, snapshot['status'])
        self.assertEqual(board.winner, snapshot['winner'])
        self.assertTrue(model.training)
        clone = board.clone()
        clone.step(4)
        self.assertEqual(board.board.ravel()[4], 0)
        self.assertEqual(board.encoder(snapshot).shape, (2, 5, 7))

    def test_immediate_win_both_players(self):
        for white in (False, True):
            with self.subTest(white=white):
                board = Chessboard(5, 5)
                moves = ([20, 0, 22, 1, 24, 2, 15, 3, 17] if white else
                         [0, 20, 1, 22, 2, 24, 3, 15])
                for action in moves:
                    board.step(action)
                planner = MCTSPlanner(UniformNet(), simulation_limit=100, seed=42)
                self.assertEqual(planner.plan(board), 4)
                self.assertEqual(planner.root.children[4].q_value, -1)

    def test_block_opponent_win(self):
        board = Chessboard(5, 5)
        for action in [0, 5, 2, 6, 4, 7, 20, 8]:
            board.step(action)
        planner = MCTSPlanner(UniformNet(), simulation_limit=800, seed=42)
        self.assertEqual(planner.plan(board), 9)

    def test_draw_terminal_without_network(self):
        board = Chessboard(1, 1)
        model = UniformNet()
        planner = MCTSPlanner(model, simulation_limit=10, seed=42)
        np.testing.assert_array_equal(planner.search(board), [1])
        self.assertEqual(model.calls, 1)
        self.assertEqual(planner.root.q_value, 0)
        board.step(0)
        with self.assertRaises(ValueError):
            planner.search(board)

    def test_noise_reproducibility_and_temperature(self):
        board = Chessboard(3, 4)
        board.step(0)
        a = MCTSPlanner(UniformNet(), simulation_limit=30, seed=17)
        b = MCTSPlanner(UniformNet(), simulation_limit=30, seed=17)
        pa = a.search(board, add_noise=True)
        np.testing.assert_array_equal(pa, b.search(board, add_noise=True))
        self.assertEqual(pa[0], 0)
        self.assertAlmostEqual(sum(c.prior for c in a.root.children.values()), 1)
        self.assertGreater(np.std([c.prior for c in a.root.children.values()]), 0)
        for temperature in (0, 1e-12, .1, 1, 10):
            pi = a.search(board, temperature=temperature)
            self.assertTrue(np.isfinite(pi).all())
            self.assertAlmostEqual(pi.sum(), 1)
            if temperature == 0:
                self.assertEqual(np.count_nonzero(pi), 1)

    def test_bad_outputs_restore_mode(self):
        class BadNet(UniformNet):
            def forward(self, x):
                p, v = super().forward(x)
                return p, v.fill_(float('nan'))
        model = BadNet().train()
        with self.assertRaises(ValueError):
            MCTSPlanner(model, simulation_limit=1).search(Chessboard(3, 3))
        self.assertTrue(model.training)
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                MCTSPlanner(UniformNet(), simulation_limit=bad)

    def test_real_model_search_and_full_game(self):
        path = Path(__file__).resolve().parents[2] / 'algorithm/networks/LeNet-5/model.py'
        spec = importlib.util.spec_from_file_location('gomoku_lenet', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(42)
        model = module.LeNet().train()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        board = Chessboard(5, 5)
        planner = MCTSPlanner(model, simulation_limit=5, seed=42)
        rng = np.random.default_rng(42)
        for _ in range(25):
            pi = planner.search(board, add_noise=True)
            action = int(rng.choice(25, p=pi))
            self.assertIn(action, board.legal_actions())
            _, done, _ = board.step(action)
            if done:
                break
        self.assertTrue(board.terminated)
        self.assertTrue(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])


if __name__ == '__main__':
    unittest.main()
