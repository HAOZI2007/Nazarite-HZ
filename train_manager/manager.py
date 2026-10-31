from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import random
from uuid import uuid4
import time

import torch

from train_manager.checkpoint import save_checkpoint, load_checkpoint

from scene.board import Chessboard
from algorithm.search.mcts.mcts import MCTSPlanner
from algorithm.data_collect.data_collector.collector import Collector
from algorithm.data_collect.sample_pool.pool import ReplayBuffer
from algorithm.training.trainer import PolicyValueTrainer

class TrainManager:
    def __init__(self, config, model_factory, *, task_id="custom"):
        self.task_id = task_id
        self.run_dir = None
        self._at_boundary = True
        self.config = deepcopy(config)
        config = self.config
        self._validate_config()

        random.seed(config.seed)
        torch.manual_seed(config.seed)

        device = ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
        self.device = torch.device(device)
        # 先移动模型，再创建优化器。
        self.model = model_factory().to(self.device)

        # 创建棋盘环境类
        self.env = Chessboard(
            config.rows,
            config.cols,
        )
        #创建mcts规划器用于自对弈产生样本
        self.planner = MCTSPlanner(
            model= self.model,
            simulation_limit= config.simulations,
            seed= config.seed,
        )
        #创建样本收集器
        self.collector = Collector(
            model=self.model,
            board=self.env,
            planner=self.planner,
            seed=config.seed,
            temperature=config.temperature,
            add_noise=config.add_noise,
        )
        #创建经验池
        self.buffer = ReplayBuffer(
            capacity=config.capacity,
            seed=config.seed,
        )
        #创建模型训练类（反向传播）
        self.trainer = PolicyValueTrainer(
            model=self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            value_weight=config.value_weight,
        )
        # completed_iterations 表示已经完整执行完的轮数。
        self.completed_iterations = 0
        self.total_games = 0
        self.total_samples = 0
        self.total_updates = 0

    def _validate_config(self):
        self.config.validate()

    def collect_games(self):
        cfg = self.config

        stats = {
            "games": 0,
            "samples": 0,
            "black_wins": 0,
            "white_wins": 0,
            "draws": 0,
        }

        started = time.perf_counter()

        for _ in range(cfg.games_per_iteration):
            samples, game_info = self.collector.collect_sample()

            # 一局结束后，所有样本已带有 z。
            self.buffer.extend(samples)

            self.total_games += 1
            self.total_samples += len(samples)

            stats["games"] += 1
            stats["samples"] += len(samples)

            winner = game_info["winner"]

            if winner == self.env.b:
                stats["black_wins"] += 1
            elif winner == self.env.w:
                stats["white_wins"] += 1
            else:
                stats["draws"] += 1

        stats["seconds"] = time.perf_counter() - started

        return stats

    def train_batches(self):
        cfg = self.config

        if len(self.buffer) < cfg.min_samples:
            return {
                "updates": 0,
                "loss": None,
                "policy_loss": None,
                "value_loss": None,
                "seconds": 0.0,
            }

        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
        }

        started = time.perf_counter()

        for _ in range(cfg.updates_per_iteration):
            batch = self.buffer.sample(
                batch_size=cfg.batch_size,
                device=self.device,
            )

            metrics = self.trainer.update(batch)

            self.total_updates += 1

            for name in totals:
                totals[name] += metrics[name]

        updates = cfg.updates_per_iteration

        return {
            "updates": updates,
            "loss": totals["loss"] / updates,
            "policy_loss": totals["policy_loss"] / updates,
            "value_loss": totals["value_loss"] / updates,
            "seconds": time.perf_counter() - started,
        }

    def run(self):
        cfg = self.config
        self._prepare_run_dir()

        print(
            f"开始训练：棋盘={cfg.rows}x{cfg.cols},"
            f"设备={self.device},"
            f"每手模拟={cfg.simulations}"
        )

        for iteration in range(
            self.completed_iterations,
            cfg.iterations,
        ):
            self._at_boundary = False
            collection = self.collect_games()
            training = self.train_batches()

            # 只有采集与更新阶段都成功结束，才记为完成一轮。
            self.completed_iterations = iteration + 1
            self._at_boundary = True
            record = {"iteration": self.completed_iterations,
                      "total_games": self.total_games, "total_samples": self.total_samples,
                      "total_updates": self.total_updates, "buffer_size": len(self.buffer),
                      "collection": collection, "training": training}
            with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if self.completed_iterations % cfg.save_interval == 0:
                self.save(self.run_dir / "latest.pt")

            if training["updates"] == 0:
                loss_text = (
                    f"暂不更新：池内 {len(self.buffer)}"
                    f"/{cfg.min_samples} 条"
                )
            else:
                loss_text = (
                    f"loss={training['loss']:.4f},"
                    f"policy={training['policy_loss']:.4f},"
                    f"value={training['value_loss']:.4f}"
                )

            print(
                f"第 {self.completed_iterations}/{cfg.iterations} 轮 | "
                f"累计对局={self.total_games} | "
                f"池内样本={len(self.buffer)} | "
                f"累计更新={self.total_updates} | "
                f"黑胜/白胜/平="
                f"{collection['black_wins']}/"
                f"{collection['white_wins']}/"
                f"{collection['draws']} | "
                f"{loss_text}"
            )

        self.save(self.run_dir / "latest.pt")
        print(f"训练完成，断点：{self.run_dir / 'latest.pt'}")
        return self.run_dir

    def _prepare_run_dir(self):
        if self.run_dir is not None:
            return
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        safe_task = ''.join(c if c.isalnum() or c in '-_' else '_' for c in self.task_id)
        self.run_dir = Path(self.config.output_dir) / safe_task / f'{stamp}-{uuid4().hex[:8]}'
        self.run_dir.mkdir(parents=True, exist_ok=False)
        metadata = {'task_id': self.task_id, 'resolved_device': str(self.device),
                    'config': asdict(self.config)}
        (self.run_dir / 'config.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    def save(self, path):
        """仅在完整轮次边界保存，避免将半轮进度错误标成已完成。"""
        if not self._at_boundary:
            raise RuntimeError('只允许在完整轮次边界保存断点')
        save_checkpoint(path, {
            'task_id': self.task_id,
            'model_class': f'{type(self.model).__module__}.{type(self.model).__qualname__}',
            'config': asdict(self.config),
            'model': self.model.state_dict(),
            'optimizer': self.trainer.optimizer.state_dict(),
            'counters': {name: getattr(self, name) for name in
                         ('completed_iterations', 'total_games', 'total_samples', 'total_updates')},
            'buffer': self.buffer.state_dict(),
            'rng': {'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    'python': random.getstate(),
                    'planner': deepcopy(self.planner.rng.bit_generator.state),
                    'collector': deepcopy(self.collector.rng.bit_generator.state)},
        })

    def restore(self, path):
        """在新建 Manager 上恢复，允许修改设备、输出目录和目标总轮数。"""
        if self.total_games or self.completed_iterations or self.run_dir is not None:
            raise RuntimeError('请在新的 Manager 上恢复断点')
        data = load_checkpoint(path)
        if data['task_id'] != self.task_id:
            raise ValueError('断点任务名与当前任务不同')
        model_class = f'{type(self.model).__module__}.{type(self.model).__qualname__}'
        if data['model_class'] != model_class:
            raise ValueError('断点模型类与当前模型不一致')
        current = asdict(self.config)
        allowed = {'iterations', 'output_dir', 'device', 'save_interval'}
        for name, value in current.items():
            if name not in allowed and data['config'].get(name) != value:
                raise ValueError(f'续训配置不一致：{name}')
        counters = data['counters']
        for name in ('completed_iterations', 'total_games', 'total_samples', 'total_updates'):
            if type(counters.get(name)) is not int or counters[name] < 0:
                raise ValueError(f'非法训练计数：{name}')
        if self.config.iterations < counters['completed_iterations']:
            raise ValueError('目标总轮数小于断点已完成轮数')
        self.buffer.load_state_dict(data['buffer'])
        if self.buffer.board_shape not in (None, (self.config.rows, self.config.cols)):
            raise ValueError('断点经验池棋盘尺寸不匹配')
        self.model.load_state_dict(data['model'])
        self.trainer.optimizer.load_state_dict(data['optimizer'])
        for name, value in counters.items():
            setattr(self, name, value)
        self.planner.rng.bit_generator.state = data['rng']['planner']
        self.collector.rng.bit_generator.state = data['rng']['collector']
        random.setstate(data['rng']['python'])
        torch.set_rng_state(data['rng']['torch'])
        cuda_states = data['rng']['cuda']
        if self.device.type == 'cuda' and len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        print(f'恢复断点：已完成 {self.completed_iterations} 轮，池内 {len(self.buffer)} 条')


# 兼容早期类名。
Training_Manager = TrainManager
