"""策略—价值网络引导的 PUCT 搜索，每次 search 从当前棋盘重新建树。

planner = MCTSPlanner(model, simulation_limit=100, seed=42)
pi = planner.search(board, temperature=1.0, add_noise=True)
action = rng.choice(board.row * board.col, p=pi)
obs, terminated, info = board.step(int(action))

search 返回完整动作分布供采样和训练；plan 返回评估用贪心动作。
搜索只在 clone 上落子，不修改真实棋盘，也不更新网络参数。
"""

import math

import numpy as np
import torch


class MCTSNode:
    """Q 从本节点待行动方视角记录，父节点选择时使用 -child.Q。

    终局节点的 player 是假设继续轮换时的下一方，仅用于价值视角。
    节点不保存整张棋盘，每次模拟在棋盘副本上还原路径。
    """

    def __init__(self, player, parent=None, action=None, prior=1.0):
        self.player = player
        self.parent = parent
        self.action = action
        self.prior = float(prior)
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def q_value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def select_child(self, exploration_constant):
        """PUCT：利用父方价值，并探索先验较高、访问较少的动作。"""
        scale = math.sqrt(max(1, self.visit_count))
        return max(
            self.children.values(),
            key=lambda child: -child.q_value
            + exploration_constant * child.prior * scale / (1 + child.visit_count),
        )

    def expand(self, actions, priors):
        for action, prior in zip(actions, priors):
            self.children[int(action)] = MCTSNode(
                -self.player, parent=self, action=int(action), prior=prior
            )

    def backup(self, value):
        """value 从本节点 player 视角给出，每退一层取反。"""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent


class MCTSPlanner:
    def __init__(self, model, simulation_limit=1000, exploration_constant=1.41,
                 device=None, dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
                 seed=None):
        if (isinstance(simulation_limit, (bool, np.bool_))
                or not isinstance(simulation_limit, (int, np.integer))
                or simulation_limit < 1):
            raise ValueError("simulation_limit 必须为正整数")
        if not math.isfinite(exploration_constant) or exploration_constant <= 0:
            raise ValueError("exploration_constant 必须为有限正数")
        if not math.isfinite(dirichlet_alpha) or dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha 必须为有限正数")
        if not math.isfinite(dirichlet_epsilon) or not 0 <= dirichlet_epsilon <= 1:
            raise ValueError("dirichlet_epsilon 必须在 [0, 1] 内")
        self.model = model
        self.simulation_limit = int(simulation_limit)
        self.exploration_constant = exploration_constant
        parameter = next(model.parameters(), None)
        self.device = torch.device(device) if device is not None else (
            parameter.device if parameter is not None else torch.device("cpu")
        )
        # 将 'cuda' 规范为具体设备编号，便于与参数设备比较。
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if any(p.device != self.device for p in model.parameters()):
            raise ValueError("模型参数与搜索设备不同，请先 model.to(device)")
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.rng = np.random.default_rng(seed)
        self.root = None  # 可查看上次搜索的访问次数、Q 和先验

    def _expand_and_evaluate(self, node, board):
        """非终局叶节点一次前向，同时获取策略先验和价值。"""
        actions = board.legal_actions()
        if not actions:
            raise ValueError("非终局棋盘没有合法动作，请检查环境状态")
        state = board.encoder(board.observation())
        inputs = torch.from_numpy(state).unsqueeze(0).to(self.device)
        logits, value = self.model(inputs)
        if logits.shape != (1, board.row * board.col) or value.numel() != 1:
            raise ValueError("网络需返回 logits (1, H*W) 和单个价值")
        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
            raise ValueError("网络输出含 NaN 或无穷值")
        scalar_value = float(value.item())
        if not -1.00001 <= scalar_value <= 1.00001:
            raise ValueError("网络价值应在 [-1, 1] 内，请检查价值头 Tanh")
        # 仅对合法动作做 softmax，非法位置完全不参与概率归一化。
        indices = torch.tensor(actions, dtype=torch.long, device=logits.device)
        priors = torch.softmax(logits[0, indices].float(), dim=0)
        priors = priors.cpu().numpy().astype(np.float64)
        priors /= priors.sum()
        # 打破同分时固定选择较小动作编号的偏置。
        order = self.rng.permutation(len(actions))
        node.expand(np.asarray(actions)[order], priors[order])
        return float(np.clip(scalar_value, -1, 1))

    def search(self, initial_state, temperature=1.0, add_noise=False):
        """返回 (H*W,) 的 float64 分布，非法位置为 0，总和为 1。

        temperature=0：选访问最多的动作；训练早期可用 1，评估用 0。
        add_noise：仅给根节点先验加入 Dirichlet 噪声，鼓励自博弈探索。
        终局禁止搜索，调用方应读取 winner 并补训练标签。
        """
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature 必须为有限非负数")
        if initial_state.terminated:
            raise ValueError("不能搜索终局棋盘，请先处理胜负或 reset()")
        root = MCTSNode(initial_state.current_player)
        self.root = root
        # 精确恢复每个子模块原本的 train/eval 状态，包括异常路径。
        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            with torch.inference_mode():
                self._expand_and_evaluate(root, initial_state)
                if add_noise:
                    noise = self.rng.dirichlet(
                        np.full(len(root.children), self.dirichlet_alpha)
                    )
                    for child, n in zip(root.children.values(), noise):
                        child.prior = ((1 - self.dirichlet_epsilon) * child.prior
                                       + self.dirichlet_epsilon * n)
                for _ in range(self.simulation_limit):
                    board = initial_state.clone()
                    node = root
                    # SELECT：已有节点沿 PUCT 下行。
                    while node.children and not board.terminated:
                        node = node.select_child(self.exploration_constant)
                        board.step(node.action)
                    if board.terminated:
                        # 终局不调用网络；node.player 是最后落子方的对手。
                        value = (0.0 if board.winner == 0 else
                                 1.0 if board.winner == node.player else -1.0)
                    else:
                        # EXPAND + EVALUATE：用网络估值，不做随机 rollout。
                        value = self._expand_and_evaluate(node, board)
                    node.backup(value)
        finally:
            for module, training in modes:
                module.training = training
        return self._visit_policy(root, initial_state.row * initial_state.col,
                                  temperature)

    def _visit_policy(self, root, action_count, temperature):
        actions = np.array(list(root.children), dtype=np.intp)
        counts = np.array([c.visit_count for c in root.children.values()],
                          dtype=np.float64)
        policy = np.zeros(action_count, dtype=np.float64)
        if temperature == 0:
            best = np.flatnonzero(counts == counts.max())
            policy[actions[self.rng.choice(best)]] = 1.0
        else:
            # 零访问动作保持零概率，对数缩放避免小温度下幂运算上溢。
            visited = counts > 0
            scaled = (np.log(counts[visited]) - np.log(counts.max())) / temperature
            weights = np.exp(scaled)
            policy[actions[visited]] = weights / weights.sum()
        return policy

    def plan(self, initial_state):
        """评估用：不加噪声，返回访问次数最多的动作编号。"""
        return int(np.argmax(self.search(initial_state, temperature=0)))
