"""运行：python -m unittest tools.test.test_train_manager -v。"""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from algorithm.networks.LeNet.model import LeNet
from train_manager.checkpoint import load_checkpoint, save_checkpoint
from train_manager.manager import TrainManager
from train_manager.registry import get_task, list_tasks, register_task
from train_manager.scripts.train import main
from train_manager.tasks.gomoku import gomoku_lenet_smoke_config
import train_manager.tasks


class TrainManagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_registry_and_cli_validation(self):
        name = 'Gomoku-LeNet-Smoke'
        self.assertIn(name, list_tasks())
        spec = get_task(name)
        first = spec.config_factory()
        first.rows = 100
        self.assertEqual(spec.config_factory().rows, 5)
        with self.assertRaises(ValueError):
            register_task(name, spec)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['--list']), 0)
        self.assertIn(name, output.getvalue())
        for args in ([], ['unknown'], [name, '--iterations', '0']):
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(args)
                self.assertEqual(error.exception.code, 2)

    def test_config_and_warmup(self):
        cfg = gomoku_lenet_smoke_config()
        for change in ({'capacity': 4}, {'batch_size': 9}, {'simulations': 0},
                       {'temperature': float('nan')}, {'seed': -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(cfg, **change).validate()
        manager = TrainManager(cfg, LeNet)
        before = {k: v.clone() for k, v in manager.model.state_dict().items()}
        self.assertEqual(manager.train_batches()['updates'], 0)
        for name, value in manager.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertEqual(manager.total_updates, 0)

    def assert_nested_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_continuous_and_resumed_training_match_exactly(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            cfg = replace(gomoku_lenet_smoke_config(), output_dir=root)
            continuous = TrainManager(cfg, LeNet)
            continuous.run()
            split = TrainManager(replace(cfg, iterations=1), LeNet)
            first_dir = split.run()
            checkpoint = first_dir / 'latest.pt'
            resumed = TrainManager(cfg, LeNet)
            resumed.restore(checkpoint)
            self.assertEqual(resumed.completed_iterations, 1)
            self.assertEqual(resumed.total_updates, 1)
            second_dir = resumed.run()
            self.assertNotEqual(first_dir, second_dir)
            for name in ('completed_iterations', 'total_games', 'total_samples', 'total_updates'):
                self.assertEqual(getattr(continuous, name), getattr(resumed, name))
            self.assertEqual(resumed.total_updates, 2)
            self.assert_nested_equal(continuous.model.state_dict(), resumed.model.state_dict())
            self.assert_nested_equal(continuous.trainer.optimizer.state_dict(),
                                     resumed.trainer.optimizer.state_dict())
            self.assert_nested_equal(continuous.buffer.state_dict(), resumed.buffer.state_dict())
            self.assert_nested_equal(continuous.buffer.sample(8), resumed.buffer.sample(8))
            self.assert_nested_equal(continuous.planner.rng.bit_generator.state,
                                     resumed.planner.rng.bit_generator.state)
            self.assert_nested_equal(continuous.collector.rng.bit_generator.state,
                                     resumed.collector.rng.bit_generator.state)
            records = [json.loads(line) for line in (second_dir / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([record['iteration'] for record in records], [2])
            metadata = json.loads((second_dir / 'config.json').read_text())
            self.assertEqual(metadata['resolved_device'], 'cpu')
            self.assertEqual(load_checkpoint(second_dir / 'latest.pt')['counters']['total_updates'], 2)
            with self.assertRaises(ValueError):
                TrainManager(cfg, LeNet, task_id='wrong-task').restore(checkpoint)
            with self.assertRaises(ValueError):
                TrainManager(replace(cfg, simulations=99), LeNet).restore(checkpoint)

    def test_atomic_checkpoint_preserves_previous_file_on_failure(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / 'latest.pt'
            target.write_bytes(b'previous checkpoint')
            with patch('train_manager.checkpoint.torch.save', side_effect=OSError('disk error')):
                with self.assertRaises(OSError):
                    save_checkpoint(target, {})
            self.assertEqual(target.read_bytes(), b'previous checkpoint')
            self.assertEqual(list(Path(root).iterdir()), [target])


if __name__ == '__main__':
    unittest.main()
