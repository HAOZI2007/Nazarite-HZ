from train_manager.config import TrainConfig


def gomoku_lenet_8x8_config():
    return TrainConfig()


def gomoku_lenet_15x15_config():
    return TrainConfig(rows=15, cols=15)


def gomoku_lenet_smoke_config():
    """小规模链路验证，不用于棋力评估。"""
    return TrainConfig(rows=5, cols=5, device='cpu', simulations=5,
                       capacity=100, min_samples=8, batch_size=8,
                       iterations=2, games_per_iteration=1,
                       updates_per_iteration=1, save_interval=1)
