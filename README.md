# Nazarite-HZ
一个从零开始的五子棋AI训练模型框架

## 网页人机对弈

在项目根目录运行，无需额外安装前端依赖：

```bash
.venv/bin/python -m play.server
```

浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)，选择训练断点并点击「加载模型」。
默认扫描 `logs/` 下的 `.pt` 文件，最新修改的模型排在最前面；也可填写本机文件路径。
加载的是本项目 LeNet 完整训练断点（`latest.pt`），棋盘行列数从断点配置读取，支持 5～25 行/列。
断点使用 Python 反序列化，仅加载自己生成的可信文件。页面加载模型会开启新一局。

- 支持执黑先行、执白后行；切换执子后点击「重新开局」生效。
- AI 复用训练中的策略价值网络和 MCTS，每步 20～1000 次搜索，可在页面调整。
- 支持悔棋（撤回你最近一手和 AI 应手）、棋谱、手数显示和胜负连线。
- 默认 CPU 推理，本地小棋盘使用单线程；服务只监听本机，不影响训练任务。
- 棋盘支持鼠标、触屏，也可聚焦后使用方向键选择、回车或空格落子。

```bash
# 启动时加载指定模型，或选择 GPU、其他端口
.venv/bin/python -m play.server --checkpoint logs/任务名/运行目录/latest.pt
.venv/bin/python -m play.server --device cuda --port 8001
# 扫描其他模型目录
.venv/bin/python -m play.server --model-dir /你的模型目录
```

重新安装项目入口后也可使用 `play` 命令；未使用项目 `.venv` 时，可在已有依赖的环境中执行
`python -m play.server`。按 `Ctrl+C` 停止服务。当前页面对接本项目 LeNet 结构，其他网络需增加对应加载适配。

## 启动训练

所有命令都在项目根目录执行：

```bash
cd /home/haozi/桌面/WUZIQI/Nazarite-HZ
uv run train --list
uv run train Gomoku-LeNet-Smoke
uv run train Gomoku-LeNet-8x8
```

`uv run` 首次运行会创建环境、安装项目依赖和 `train` 命令。需要 Python 3.10 或以上。
当前机器已创建复用 `wuziqi` 依赖的 `.venv`，并安装项目入口；可以使用
`uv run --no-sync train Gomoku-LeNet-Smoke` 直接运行，跳过联网同步。
`--no-sync` 仅适用于依赖和项目入口已经安装好的环境。
不使用 uv 时，也可以在已安装依赖的环境中运行
`python -m train_manager.scripts.train Gomoku-LeNet-Smoke`。

| 任务名 | 默认行为 |
| --- | --- |
| `Gomoku-LeNet-Smoke` | CPU，5×5，2 轮，每轮 1 局、1 次更新，每手 5 次模拟；验证链路 |
| `Gomoku-LeNet-8x8` | 8×8，100 轮，每轮 5 局、20 次更新，每手 100 次模拟 |
| `Gomoku-LeNet-15x15` | 15×15，其余默认训练参数与 8×8 相同 |

Smoke 的目的只是验证流程，不代表已经训练出可用棋力。
正式任务默认 `device='auto'`，CUDA 可用时使用 GPU，否则使用 CPU。
MCTS 与棋盘操作仍由 CPU 执行，GPU 用于网络计算。

常用覆盖参数：

```bash
uv run train Gomoku-LeNet-8x8 --iterations 200 --device cuda --output-dir logs
```

`--iterations` 始终是目标**总轮数**。每轮先采集若干完整对局，再从经验池抽样更新网络。
当池内样本不足 `min_samples` 时仅采集，暂不更新。
容量 `capacity`、阈值 `min_samples` 和 `batch_size` 都以样本条数（手数）计，
必须满足 `capacity >= min_samples >= batch_size`。

## 运行输出与断点续训

每次启动都会新建独立目录，包括恢复训练：

```text
logs/Gomoku-LeNet-8x8/时间戳-唯一编号/
├── config.json      # 任务名、配置和实际设备
├── metrics.jsonl    # 每行记录一个完整轮次的采集、损失和累计计数
└── latest.pt        # 最近保存的完整训练断点
```

断点包含网络、Adam 优化器、完整经验池、累计轮次/对局/样本/更新计数，以及随机数状态。
默认每 10 轮保存一次，训练正常结束时也会保存。Smoke 每轮保存。
文件先写入同目录临时文件，再替换 `latest.pt`，避免写入失败时破坏原断点。
中途终止时可以从最近一次已保存的完整轮次重跑；不会自动保存半轮状态。

```bash
uv run train Gomoku-LeNet-8x8 \
  --resume logs/Gomoku-LeNet-8x8/实际运行目录/latest.pt \
  --iterations 200
```

例如断点已完成 100 轮，目标设为 200 表示继续执行第 101～200 轮。
CLI 从断点读取原配置，只允许覆盖目标总轮数、设备和输出目录。
恢复时会校验任务、模型类、关键训练配置和经验池尺寸。
只加载自己生成、可信的断点；这里保存的是完整训练状态，不是单独的权重文件。
相同环境的 CPU 续训已验证与连续训练逐项一致，跨设备不承诺逐位复现。

## 代码分工与新增任务

```text
scripts/train.py → tasks/__init__.py → registry.get_task(任务名)
                  → 配置工厂 + 模型工厂 + TrainManager
                  → Collector/MCTS → ReplayBuffer → Trainer → checkpoint
```

- `train_manager/config.py`：配置字段、默认值和参数校验。
- `train_manager/registry.py`：保存任务名到 `TaskSpec` 的映射，检查重复注册与未知任务。
- `train_manager/tasks/gomoku.py`：各任务的配置工厂，每次调用返回独立配置。
- `train_manager/tasks/__init__.py`：显式注册内置任务，不会启动训练。
- `train_manager/manager.py`：创建共享网络及各组件，组织采集、更新、日志和保存/恢复。
- `train_manager/checkpoint.py`：断点格式、原子保存和加载。
- `train_manager/scripts/train.py`：命令行解析及任务启动。

新增任务时先在 `tasks` 中定义配置工厂，再在 `tasks/__init__.py` 注册，例如：

```python
def my_config():
    return TrainConfig(rows=10, cols=10, simulations=200)

register_task('Gomoku-LeNet-10x10', TaskSpec(my_config, LeNet, TrainManager))
```

随后即可用 `uv run train Gomoku-LeNet-10x10` 启动。
配置字段统一为 `rows`、`cols`、`iterations`；旧类名 `Training_Manager` 保留为兼容别名。
管理器不负责损失函数与反向传播，这部分仍在 `algorithm/training/trainer.py`。

## 验证

```bash
python -m unittest tools.test.test_train_manager tools.test.test_player tools.test.test_mcts tools.test.test_collector tools.test.test_pool tools.test.test_trainer -v
```

覆盖合法对弈、MCTS、采集、经验池、真实参数更新、任务入口、配置校验、断点原子保存，
以及连续两轮与分段续训的模型/优化器/经验池/随机数状态一致性。
