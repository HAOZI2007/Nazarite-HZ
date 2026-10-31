"""模型对弈接口：加载、合法应手、悔棋重建和终局保护。"""

from pathlib import Path
import tempfile
import unittest

import torch

from algorithm.networks.LeNet.model import LeNet
from play.server import PlayService
from scene.board import InvalidActionError
from train_manager.checkpoint import save_checkpoint


class PlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'latest.pt'
        save_checkpoint(self.path, {
            'task_id': 'test', 'model_class': f'{LeNet.__module__}.{LeNet.__qualname__}',
            'config': {'rows': 8, 'cols': 8}, 'model': LeNet().state_dict(),
            'counters': {'completed_iterations': 12}, 'optimizer': {}, 'buffer': {}, 'rng': {},
        })
        self.service = PlayService(self.directory.name)

    def test_load_and_model_list(self):
        self.assertEqual(self.service.models()[0]['path'], str(self.path))
        info = self.service.load(str(self.path))
        self.assertEqual((info['rows'], info['cols'], info['iterations']), (8, 8, 12))

    def test_ai_plays_legal_move_for_both_colors(self):
        self.service.load(str(self.path))
        first = self.service.move([], 5)
        second = self.service.move(first['moves'], 5)
        self.assertNotEqual(first['action'], second['action'])
        self.assertEqual(second['current_player'], -1)
        self.assertEqual(len(second['moves']), 2)
        self.assertEqual(sum(v != 0 for row in second['board'] for v in row), 2)

    def test_rebuild_for_undo(self):
        self.service.load(str(self.path))
        self.service.move([27], 5)
        empty = self.service.position([])
        self.assertEqual(empty['current_player'], -1)
        self.assertEqual(sum(map(sum, empty['board'])), 0)

    def test_illegal_moves_and_input(self):
        self.service.load(str(self.path))
        for moves in ([0, 0], [-1], [64], [True], [1.5]):
            with self.subTest(moves=moves), self.assertRaises(InvalidActionError):
                self.service.position(moves)
        for count in (0, 2001, True, '100'):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.service.move([], count)

    def test_win_and_terminal_protection(self):
        self.service.load(str(self.path))
        moves = [0, 8, 1, 9, 2, 10, 3, 11, 4]
        result = self.service.position(moves)
        self.assertEqual(result['winner'], -1)
        self.assertTrue(result['terminated'])
        self.assertIsNone(result['current_player'])
        with self.assertRaises(ValueError):
            self.service.move(moves, 5)
        with self.assertRaises(InvalidActionError):
            self.service.position([*moves, 12])

    def test_failed_load_keeps_previous_model(self):
        info = self.service.load(str(self.path))
        with self.assertRaises(ValueError):
            self.service.load(str(self.path.parent / 'missing.pt'))
        self.assertEqual(self.service.info, info)
        self.assertEqual(len(self.service.move([], 1)['moves']), 1)

    def test_model_required(self):
        with self.assertRaisesRegex(ValueError, '先加载'):
            self.service.move([], 5)


if __name__ == '__main__':
    unittest.main()
