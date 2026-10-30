# MCTS 与自博弈训练笔记

> 整理日期：2026-10-30。以本项目当前 `MCTSPlanner`、`MCTSNode` 和 `Chessboard` 为准。
> 本文只讲 MCTS、自博弈采样与训练衔接，不展开卷积网络结构。

## 1. 先区分五个概念

| 概念 | 含义 |
| --- | --- |
| 一次模拟 simulation | 从当前局面副本出发，沿搜索树试走一条路径，评估叶节点，再回传 |
| 一次搜索 search | 在同一真实局面进行多次模拟，生成搜索策略 π |
| 一手真实落子 | 根据 π 采样动作，对真实环境执行一次 step |
| 一局自博弈 | 黑白交替执行“搜索 → 真实落子”，直到获胜或平局 |
| 一轮训练迭代 | 收集若干局样本，再抽取若干 batch 更新网络 |

当前实现采用网络引导的 PUCT 搜索：策略头给搜索先验，价值头评估非终局叶节点。它不做随机下到终局的 rollout。

网络在第一局自博弈前就要创建，但权重可以是随机的。前向计算不要求网络已经训练好；只是初始预测通常没有棋力。

- 自博弈搜索：使用网络前向计算，不更新参数。
- 网络训练：使用自博弈样本计算损失，反向传播，更新参数。
- MCTS 自身不一定需要网络，但本项目这个 AlphaZero 风格版本需要。

## 2. 当前代码与接口

路径相对于 `Nazarite-HZ/`：

| 文件 | 作用 |
| --- | --- |
| `algorithm/search/mcts/mcts.py` | 节点、PUCT、模拟、回传和策略生成 |
| `scene/board.py` | 真实与模拟棋盘使用的规则接口 |
| `tools/test/test_mcts.py` | MCTS 行为测试 |

关键函数：

| 函数 | 输入 | 输出或效果 |
| --- | --- | --- |
| `env.reset()` | 无 | 重置真实对局，返回 obs |
| `env.observation()` | 无 | 包含棋盘副本、行动方、赢家等的字典 |
| `env.encoder(obs)` | 非终局观测 | `(2,H,W)` 的 float32 数组 |
| `env.clone()` | 无 | 独立棋盘副本 |
| `board.legal_actions()` | 无 | 合法整数动作列表 |
| `board.step(action)` | 整数动作 | `(obs, terminated, info)`，共三项 |
| `planner.search(env, temperature, add_noise)` | 当前真实环境及搜索选项 | `(H*W,)` 的搜索概率分布 π |
| `planner.plan(env)` | 当前真实环境 | 访问次数最多的动作编号 |
| `_expand_and_evaluate(node, board)` | 节点和对应局面的环境 | 展开节点，返回该节点行动方视角的网络价值 |
| `node.expand(actions, priors)` | 合法动作及网络先验 | 创建子节点 |
| `node.select_child(c)` | 探索常数 | 返回 PUCT 分数最高的子节点 |
| `node.backup(value)` | 当前节点视角的价值 | 沿路径更新 N、W，每层取反 |
| `_visit_policy(root, action_count, temperature)` | 根节点统计 | 将访问次数转换成 π |

当前棋子约定：黑方 `-1`，白方 `+1`，空位 `0`。动作编码：

```python
action = row * 棋盘列数 + col
row, col = divmod(action, 棋盘列数)
```

终局的 `obs['current_player']` 为 `None`，不能直接送入 encoder。

## 3. 真实棋盘、模拟棋盘与搜索树

```text
真实环境 env
    │
    ├─ clone() → 模拟棋盘 1 → 试走 → 丢弃
    ├─ clone() → 模拟棋盘 2 → 试走 → 丢弃
    └─ clone() → 模拟棋盘 3 → 试走 → 丢弃

这些模拟共享本次 search 的搜索树及统计。
真实环境在整个 search 中保持不变。
```

搜索树节点没有保存完整棋盘。每次模拟从真实棋盘副本开始，沿路径执行动作，将副本推进到对应节点的局面。

同一次 search 中：棋盘副本不断重建，搜索树不断积累。

不同真实落子之间：当前实现重新建树，不复用上一次搜索树。`planner.root` 只保留最近一次搜索的根，方便查看统计。

## 4. 节点里保存什么

```python
class MCTSNode:
    def __init__(self, player, parent=None, action=None, prior=1.0):
        self.player = player
        self.parent = parent
        self.action = action
        self.prior = float(prior)
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
```

| 字段 | 含义 |
| --- | --- |
| `player` | 到达这个节点后，轮到谁行动 |
| `parent` | 父节点引用 |
| `action` | 从父节点到这里需要执行的动作 |
| `prior` / P | 父节点网络对这个动作给出的先验概率 |
| `children` | 动作编号到子节点的映射 |
| `visit_count` / N | 搜索经过该节点的次数 |
| `value_sum` / W | 从本节点 player 视角累计的价值 |
| `q_value` / Q | W / N；未访问时暂取 0 |

最重要的视角约定：

> Q 属于本节点的待行动方；父节点评价这个子节点时需要取负号。

例子：父节点黑方行动，黑方落子 A 后到达白方节点。若该白方节点 Q 为 `-0.6`，意味着对白方不利，因此黑方看动作 A 的价值为 `+0.6`。

## 5. 一次 search 的完整函数调用顺序

```text
planner.search(env)
│
├─ MCTSNode(env.current_player)        创建根节点
├─ _expand_and_evaluate(root, env)     根节点初始化
│  ├─ board.legal_actions()
│  ├─ board.observation()
│  ├─ board.encoder(obs)
│  ├─ model(inputs)                   网络前向
│  ├─ 对合法 logits 做 softmax
│  └─ node.expand(actions, priors)
│
├─ 可选：根节点先验加入 Dirichlet 噪声
│
├─ 重复 simulation_limit 次
│  ├─ board = env.clone()
│  ├─ node = root
│  ├─ while 节点已展开且棋盘未终局
│  │  ├─ node.select_child(c)
│  │  └─ board.step(node.action)      只在副本落子
│  ├─ 若终局：根据 winner 计算价值
│  │  否则：_expand_and_evaluate(node, board)
│  └─ node.backup(value)
│
└─ _visit_policy(...)                 返回搜索策略 π
```

### 5.1 根节点初始化：网络同时输出 P 与 v

`_expand_and_evaluate()` 的主要逻辑：

```python
actions = board.legal_actions()
state = board.encoder(board.observation())
inputs = torch.from_numpy(state).unsqueeze(0).to(self.device)
logits, value = self.model(inputs)
```

数据形状：

```text
obs：环境状态字典
 ↓ encoder
state：(2,H,W)，当前玩家与对手两层
 ↓ from_numpy + unsqueeze(0)
inputs：(1,2,H,W)
 ↓ 网络
logits：(1,H*W)
value：一个当前行动方视角的数
```

仅提取合法位置，计算先验：

```python
indices = torch.tensor(actions, dtype=torch.long, device=logits.device)
priors = torch.softmax(logits[0, indices].float(), dim=0)
```

例如非法位置分数再高，也不会参与归一化或创建子节点。

当前函数会检查输出形状、有限值以及价值是否在 `[-1,1]`，避免错误预测污染树统计。

### 5.2 expand：创建候选分支，不实际试走

```python
node.expand(actions, priors)
```

子节点来自两种信息：

- 环境给出合法动作 actions。
- 网络给出对应动作的先验 priors。

每个子节点记录动作、先验和相反的行动方，统计初始化为零。这个步骤不调用 step，不逐个评估所有子节点。

网络价值 v 不用于创建动作分支，而是用于本次模拟的回传。根节点初始化是例外：当前实现只使用其策略输出，忽略根节点价值，不将这次初始化计入模拟访问次数。

代码随机排列动作和对应先验，是为了在 PUCT 完全同分时不总选择最小动作编号；没有改变动作与概率的对应关系。

### 5.3 select_child：PUCT 挑选值得探索的分支

当前实现：

```python
score = (
    -child.q_value
    + c * child.prior
      * math.sqrt(max(1, parent.visit_count))
      / (1 + child.visit_count)
)
```

对应公式：

```text
score = -Q_child + c × P_child × sqrt(max(1, N_parent)) / (1 + N_child)
```

- `-Q_child`：把子节点价值转换成父方视角，利用已有评估。
- P 较大：网络认为值得尝试。
- 子节点 N 较小：尚未充分探索，探索项较大。
- c：探索强度。
- `max(1,N_parent)`：让零访问时也有有效探索项。

PUCT 同时考虑 P、Q、N，并不是直接选择网络概率最大的动作。

### 5.4 模拟中的 step 与真实 step

```python
board = env.clone()
node = root

while node.children and not board.terminated:
    node = node.select_child(self.exploration_constant)
    board.step(node.action)
```

每选择一个子节点，模拟棋盘也执行一步，保证“树节点”和“模拟局面”同步。

```python
board.step(...)  # 搜索内部：副本试走

env.step(...)    # 搜索外部：真实对弈落子
```

循环遇到未展开节点或终局时停止，因此搜索深度不固定。一次模拟可能只走一手，也可能沿已有树深入多手。

### 5.5 非终局评估与终局评估

非终局：

```python
value = self._expand_and_evaluate(node, board)
```

调用网络，展开这个叶节点，然后本次模拟停止，不继续随机下完整盘。

终局：

```python
value = (
    0.0 if board.winner == 0 else
    1.0 if board.winner == node.player else
    -1.0
)
```

终局无需网络预测。即使环境观测不再提供当前玩家，树节点仍保留按黑白交替推导出的 player，用于统一价值视角。

例如黑方刚成五：终局子节点 player 为白方，因此该节点价值为 `-1`。这不表示白方还要继续行动，只是用于计算。

### 5.6 backup：回传价值，每层取反

```python
def backup(self, value):
    node = self
    while node is not None:
        node.visit_count += 1
        node.value_sum += value
        value = -value
        node = node.parent
```

例如：

```text
根：黑方行动
 ↓ 黑方下 A
节点 A：白方行动
 ↓ 白方下 D
叶节点 D：黑方行动，网络价值 +0.6
```

回传：

| 节点 | 视角 | 本次加到 W 的值 |
| --- | --- | ---: |
| D | 黑方 | +0.6 |
| A | 白方 | -0.6 |
| 根 | 黑方 | +0.6 |

回传取反让每个节点存自己的价值；选择时使用 `-child.Q` 则是将读取到的子方价值转成父方价值。两者不是多余的重复操作。

## 6. 连续三次模拟：不跳步的数值例子

以下为手算教学例子，不是运行日志。假设只考虑 A、B、C 三个候选，或其他候选在这几次模拟中未被选中。真实代码仍保留全部合法动作。

条件：当前真实局面 S0，黑方行动；关闭噪声；c=1；仅做三次模拟。

根网络给出先验：

```text
P(A)=0.6，P(B)=0.3，P(C)=0.1
```

展开根后，各子节点 N、W、Q 都是零，子节点 player 都是白方。

### 6.1 第一次模拟

1. `board = env.clone()`：得到 S0 副本。
2. `root.select_child(1)`：零访问时按先验，选 A。
3. `board.step(A)`：副本变成 SA，轮到白方。
4. SA 未终局，A 尚未展开。
5. `_expand_and_evaluate(A, board)`：网络按白方视角评估，假设 v=+0.4。
6. `A.expand(...)`：创建白方应对分支。
7. `A.backup(+0.4)`：A 加 +0.4，根加 -0.4，各自 N 加一。
8. 丢弃副本，保留树统计。真实 env 仍为 S0。

```text
根：N=1，W=-0.4
A：N=1，W=+0.4，Q=+0.4
B：N=0，Q=0
C：N=0，Q=0
```

### 6.2 第二次模拟

1. 再次 `env.clone()`，从 S0 开始，沿用已有树。
2. 计算根节点 PUCT：

```text
A：-0.4 + 0.6 × 1 / 2 = -0.1
B： 0   + 0.3 × 1 / 1 =  0.3
C： 0   + 0.1 × 1 / 1 =  0.1
```

3. 选 B；在副本 `board.step(B)`，到达白方行动的 SB。
4. SB 未终局，B 未展开。
5. `_expand_and_evaluate(B, board)`：假设白方 v=-0.2，白方应对 D、E 的先验为 0.7、0.3。
6. `B.expand(...)` 创建 D、E；它们 player 为黑方，N 为零。
7. `B.backup(-0.2)`：B 加 -0.2，根加 +0.2。
8. 丢弃副本，真实 env 不变。

```text
根：N=2，W=-0.2
A：N=1，Q=+0.4
B：N=1，W=-0.2，Q=-0.2
C：N=0，Q=0
```

### 6.3 第三次模拟

1. 再次克隆 S0。
2. 根节点 PUCT：

```text
A：-0.4 + 0.6 × √2 / 2 ≈ 0.024
B：+0.2 + 0.3 × √2 / 2 ≈ 0.412
C： 0   + 0.1 × √2 / 1 ≈ 0.141
```

3. 选 B，副本执行黑方 B，到达 SB。
4. B 已经展开，while 不停止，继续调用 `B.select_child(1)`。
5. D、E 未访问，D 的先验较大，选 D。
6. 副本执行白方 D，得到 SBD，轮到黑方。
7. SBD 未终局，D 未展开。
8. `_expand_and_evaluate(D, board)`：假设网络预测黑方 v=-0.6，并展开 D。
9. `D.backup(-0.6)`：D 加 -0.6，B 加 +0.6，根加 -0.6。
10. 丢弃副本。

```text
根：N=3，W=-0.8，Q≈-0.267
A：N=1，W=+0.4，Q=+0.4
B：N=2，W=+0.4，Q=+0.2
C：N=0，Q=0
D：N=1，W=-0.6，Q=-0.6
```

B 第一次的白方估计为 -0.2，深入看到白方应对 D 后回传 +0.6，因此其白方平均价值修正为 +0.2。搜索结果可能随探索深入改变。

## 7. 从访问次数生成搜索策略 π

函数：`_visit_policy()`。

以上例子根子节点访问为：A=1、B=2、C=0。温度 τ=1 时：

```text
π(A)=1/3，π(B)=2/3，π(C)=0
```

一般公式：

```text
π(a) = N(a)^(1/τ) / Σ_b N(b)^(1/τ)
```

| 温度 | 效果 |
| --- | --- |
| τ=1 | 访问次数按比例归一化 |
| 0<τ<1 | 更集中于高访问动作 |
| τ>1 | 更平缓 |
| τ=0 | 选访问最多的动作，返回 one-hot；并列时随机打破平局 |

实现用对数缩放，避免小温度下幂运算上溢；零访问动作保持零概率。

返回完整 `(H*W,)` 数组，非法位置为零，总和为一。当前返回 dtype 为 float64，训练组 batch 时可转换成 float32。

网络先验 P 与最终策略 π 不相同：P 是搜索起点，π 是搜索之后的访问分布。

搜索预算太小时，π 可能非常稀疏。例如 225 个候选只模拟 10 次，最多访问 10 个根动作。搜索并不保证每次都找到最佳动作。

## 8. 根噪声、温度和随机数各自做什么

根噪声仅在 `add_noise=True` 时使用：

```text
P' = (1-ε)P + εη
η ~ Dirichlet(α)
```

默认 α=0.3、ε=0.25，即混合 75% 网络先验与 25% 随机分布。

- 根噪声：改变根节点搜索先验，影响探索方向。
- 温度：搜索结束后将访问次数转换为动作分布。
- `rng.choice(..., p=pi)`：最终从这个分布采样真实动作。

它们是三个不同步骤。ε=0.25 不表示有 25% 概率直接随机落子。

评估通常使用 `planner.plan(env)`：不加噪声，温度为零。

`seed` 控制搜索器内部随机顺序、噪声和并列选择；真实动作采样使用的 rng 也需要单独设 seed。网络初始化还需要 torch 的 seed。相同 seed 之外，模型、输入及调用顺序也要相同，才便于复现。

## 9. search 结束后才发生真实落子

```python
obs = env.observation()
player = obs['current_player']
state = env.encoder(obs)

pi = planner.search(env, temperature=1.0, add_noise=True)

trajectory.append((state.copy(), pi.copy(), player))
action = int(rng.choice(env.row * env.col, p=pi))
obs, terminated, info = env.step(action)
```

保存的是落子前状态，不是 step 后的状态。即使本次采样选 B，仍保存完整 π，而非将 B 改成 one-hot 标签。

模拟中访问过的 SA、SB、SBD 不会自动成为训练样本。本项目这套基础采样方案只保存每一手真实行动前的根局面。

下一手白方会从真实落子后的局面重新建树、搜索、保存和采样。上一轮模拟试过白方 D，不代表真实白方必须走 D。

## 10. 完整自博弈采样示例

以下是建议的采集函数示例，不代表当前仓库已经实现经验池或完整训练器。env、planner、rng 由外部创建。

```python
def collect_one_game(env, planner, rng):
    obs = env.reset()
    trajectory = []

    while not obs['terminated']:
        player = obs['current_player']
        state = env.encoder(obs)

        pi = planner.search(
            env,
            temperature=1.0,
            add_noise=True,
        )

        trajectory.append((state.copy(), pi.copy(), player))
        action = int(rng.choice(env.row * env.col, p=pi))
        obs, terminated, info = env.step(action)

    winner = obs['winner']
    samples = []
    for state, pi, player in trajectory:
        if winner == 0:
            z = 0.0
        elif winner == player:
            z = 1.0
        else:
            z = -1.0
        samples.append((state, pi, z))

    return samples
```

### 10.1 终局后补 z 是标注真实结果，不是逐手打好坏分

黑方最终获胜：黑方所有记录标 +1，白方所有记录标 -1。平局全部标 0。

这不表示赢家每一步都好，也不表示输家每一步都坏。z 是该次实际对局的最终结果；跨多局积累后，网络学习预期结果。

当前 step 不返回 reward，调用方根据 winner 和记录的 player 生成 z，不必为活三、冲四另设奖励。

| 数据 | 来源 | 用途 |
| --- | --- | --- |
| P | 网络策略输出 | 搜索先验 |
| v | 网络价值输出 | 非终局叶节点评估 |
| Q | 多次模拟回传的平均值 | PUCT 选择 |
| π | 根动作访问次数 | 真实动作采样、策略训练目标 |
| z | 真实整局结果，按记录玩家视角 | 价值训练目标 |

不能用搜索中的 Q 或 v 冒充真实终局标签 z。

### 10.2 一局九手，产生九条原始样本

5×5 教学对局，坐标从零开始：

| 手数 | 玩家 | 落点 | 动作编号 | 黑方最终获胜后的 z |
| --- | --- | --- | ---: | ---: |
| 1 | 黑 | (0,0) | 0 | +1 |
| 2 | 白 | (1,0) | 5 | -1 |
| 3 | 黑 | (0,1) | 1 | +1 |
| 4 | 白 | (1,1) | 6 | -1 |
| 5 | 黑 | (0,2) | 2 | +1 |
| 6 | 白 | (1,2) | 7 | -1 |
| 7 | 黑 | (0,3) | 3 | +1 |
| 8 | 白 | (1,3) | 8 | -1 |
| 9 | 黑 | (0,4) | 4 | +1 |

第九手输入是已有八颗棋子的局面，不是黑方成五后的终局棋盘。

一局产生的原始样本数等于真实记录的手数，不等于 MCTS 模拟次数或搜索节点数。没有额外记录开局或数据增强时，一手对应一条。

## 11. 经验池与网络更新

经验池通常按样本条数计数，不是按整局数计数。一局平均 40 手，10 局约有 400 条原始样本。

不必等经验池填满，达到最小样本量且足够抽 batch 后即可训练。样本可以来自不同棋局、不同手数，不要求时间连续。

一批样本形状：

```text
states：         (B,2,H,W)，float32
target_policy：  (B,H*W)，float32，每行非负且和为 1
target_outcome： (B,)，float32，取 -1、0、+1
```

同一 batch 的棋盘尺寸要一致。当前基础实现可以先固定棋盘大小；以后多尺寸训练可按尺寸分组采样。

训练使用当前网络重新预测，不需要保存采集时的旧网络 p、v。

```python
policy_logits, predicted_value = model(states)
```

策略损失与价值损失：

```text
L_policy = -mean_batch(Σ_a π(a) × log p(a))
L_value  = mean_batch((v-z)^2)
L_total  = L_policy + λ × L_value
```

λ 可以先取 1。优化器示例：

```python
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3, weight_decay=1e-4
)
```

创建模型并移动到目标设备后，再创建优化器；整个训练过程复用优化器，不要每个 batch 重建。

单次更新示例：

```python
import torch
import torch.nn.functional as F


def train_one_batch(model, optimizer, batch):
    # batch 由调用方准备好：float32，且与模型同设备。
    states, target_policy, target_outcome = batch
    model.train()
    optimizer.zero_grad()
    logits, values = model(states)

    policy_loss = -(
        target_policy * F.log_softmax(logits, dim=1)
    ).sum(dim=1).mean()
    value_loss = F.mse_loss(values.squeeze(-1), target_outcome)
    loss = policy_loss + value_loss

    loss.backward()
    optimizer.step()
    return loss.item(), policy_loss.item(), value_loss.item()
```

这里策略损失对全位置 logits 归一化，非法位置目标为零。实际搜索和选动作始终屏蔽非法位置。若改为训练时也掩码，需要小心 `0 × -inf` 产生 NaN。

价值输出 `(B,1)` 要用 `squeeze(-1)` 对齐标签 `(B,)`，避免广播出错误形状。

采用 optimizer 的 weight_decay 后，起步版本不必再手工重复添加 L2 项。

共享卷积参数接收两种损失的梯度；策略头参数只接收策略损失梯度，价值头参数只接收价值损失梯度。两者通过共享特征间接影响彼此。

## 12. 多轮自博弈训练如何衔接

第一版建议采用串行流程：采集期间固定参数，采集完成后再训练。

```text
创建随机网络 θ0
    ↓
用 θ0 完成若干局 MCTS 自博弈
    ↓
得到 (state, π, z)，加入经验池
    ↓
抽 batch 更新网络，得到 θ1
    ↓
用 θ1 进行下一批自博弈
    ↓
新样本加入经验池，从新旧样本混合抽样训练
```

伪代码，经验池接口需要另外实现：

```python
for iteration in range(num_iterations):
    model.eval()
    for _ in range(games_per_iteration):
        samples = collect_one_game(env, planner, rng)
        replay_buffer.extend(samples)

    if len(replay_buffer) >= min_samples:
        for _ in range(updates_per_iteration):
            batch = replay_buffer.sample(batch_size)
            train_one_batch(model, optimizer, batch)
```

planner 保存的是同一个 model 对象，因此 optimizer 原地更新参数后，下一次 search 自动使用新权重。若另外创建了新的 model 对象，则需要重新绑定 planner 或创建新 planner。

例如每轮采集 10 局、训练 20 步只是流程示例，并非推荐固定超参数。样本不足 batch_size 时不能直接抽样。

训练不保证每轮棋力都提升。评估应固定搜索预算、交换先后手，与固定旧模型或其他固定基准对局，不能只看 loss。

## 13. 搜索时的推理模式

当前 search 使用：

```python
model.eval()
with torch.inference_mode():
    ...
```

- eval 控制 Dropout、BatchNorm 等模块行为，不等于关闭梯度。
- inference_mode 关闭梯度记录等推理开销，不更新参数。
- search 结束或报错时恢复各模块原来的 train/eval 状态。
- 网络参数更新只发生在训练代码的 backward、optimizer.step 中。

当前实现适合串行采集与训练，不应让另一个线程在 search 中途更新同一个 model。以后并发采集可以使用独立的推理模型快照。

## 14. 容易混淆的地方

| 容易误解 | 正确理解 |
| --- | --- |
| expand 使用网络 value 创建候选点 | 合法动作来自环境，先验来自策略头；value 用于回传 |
| expand 会把所有候选动作都下出来 | 只创建节点，实际试走由模拟中的 step 执行 |
| 每次模拟从上次模拟结束的棋盘继续 | 每次重新克隆同一个真实局面，但保留搜索树 |
| PUCT 就是按网络概率选动作 | 同时考虑 P、Q、N |
| 模拟一定要下到终局 | 本实现遇到新叶节点就用价值网络评估并停止 |
| 每访问一个节点就得到一条训练样本 | 基础采集器只记录真实落子前的根状态 |
| 搜索回传就是反向传播 | backup 更新树的 N、W；backward 更新网络梯度，两者不同 |
| v、Q、z 都是一个分数 | 分别是网络预测、搜索统计、真实终局标签 |
| 黑方永远正价值、白方永远负价值 | 价值从当前节点玩家视角表示，黑白都能取正负 |
| 终局标签是在逐手判断好坏 | 它记录这次真实对局的最终胜负 |
| 第一局网络未训练，不能用于 MCTS | 可以前向，但初始质量有限，纯自博弈启动较慢 |
| search 已经执行真实动作 | search 只返回 π；真实 step 在外部调用 |

## 15. 当前实现范围与检查方式

已包含：网络先验、PUCT、棋盘副本模拟、真实终局价值、逐层取反回传、访问策略、温度、根噪声、输入输出检查及推理模式恢复。

尚未包含：跨真实落子的树复用、批量叶节点推理、并发采集、完整经验池、训练器及评估器。本文中的采集和训练片段是后续搭建参考。

当前网络和搜索较简单，有限搜索预算及弱价值预测都可能导致漏看战术；正确实现不等于已拥有强棋力。

从 `Nazarite-HZ/` 目录使用安装了依赖的 Python 环境运行：

```bash
python -m unittest tools.test.test_mcts -v
python -m tools.test.test_player --games 100 --seed 42
```

阅读源码建议顺序：

```text
search
 → _expand_and_evaluate
 → expand
 → select_child
 → backup
 → _visit_policy
 → 外部自博弈采样循环
 → 经验池与 train_one_batch
```
