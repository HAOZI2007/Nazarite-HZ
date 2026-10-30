import math
import time

import torch

from scene.board import Chessboard
from algorithm.search.mcts.mcts import MCTSPlanner
from algorithm.data_collect.data_collector.collector import Collector
from algorithm.data_collect.sample_pool.pool import ReplayBuffer
from algorithm.training.trainer import PolicyValueTrainer

class Training_Manager:
    def __init__(self,config,model_factory):
        self.config = config
        self._validate_config()

        torch.manual_seed(config.seed)

        self.device = torch.device(config.device)
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
        #穿境模型训练类（反向传播）
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
        cfg = self.config

        positive_integer_fields = (
            "rows",
            "cols",
            "simulations",
            "capacity",
            "min_samples",
            "batch_size",
            "iterations",
            "games_per_iteration",
            "updates_per_iteration",
        )

        for name in positive_integer_fields:
            value = getattr(cfg, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} 必须为正整数")

        if not (
            cfg.capacity >= cfg.min_samples >= cfg.batch_size
        ):
            raise ValueError(
                "必须满足 capacity >= min_samples >= batch_size"
            )

        if (
            not math.isfinite(cfg.learning_rate)
            or cfg.learning_rate <= 0
        ):
            raise ValueError("learning_rate 必须为有限正数")

        for name in ("weight_decay", "value_weight", "temperature"):
            value = getattr(cfg, name)

            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须为有限非负数")

        if not isinstance(cfg.add_noise, bool):
            raise ValueError("add_noise 必须为布尔值")

        if (
            isinstance(cfg.seed, bool)
            or not isinstance(cfg.seed, int)
            or cfg.seed < 0
        ):
            raise ValueError("seed 必须为非负整数")

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

        print(
            f"开始训练：棋盘={cfg.rows}x{cfg.cols},"
            f"设备={self.device},"
            f"每手模拟={cfg.simulations}"
        )

        for iteration in range(
            self.completed_iterations,
            cfg.iterations,
        ):
            collection = self.collect_games()
            training = self.train_batches()

            # 只有采集与更新阶段都成功结束，才记为完成一轮。
            self.completed_iterations = iteration + 1

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