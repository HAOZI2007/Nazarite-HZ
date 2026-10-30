from dataclasses import dataclass
import torch

@dataclass
class TrainConfig:
    row: int = 8
    col: int = 8
    seed: int = 42
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    simulations: int = 100
    temperature: float = 1.0
    add_noise: bool = True

    capacity: int = 10000
    min_samples: int = 128
    batch_size: int = 32

    max_iterations: int = 100
    games_per_iteration: int = 5
    updates_per_iteration: int = 20

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0

    output_dir: str = "logs"
    save_interval: int = 10