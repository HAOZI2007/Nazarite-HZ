import numpy as np
from enum import Enum


class StepErrorCode(Enum):
    GAME_OVER = "game_over"
    INVALID_TYPE = "invalid_type"
    OUT_OF_BOUNDS = "out_of_bounds"
    OCCUPIED = "occupied"


class InvalidActionError(ValueError):
    """非法落子;code 可供调用方判断，消息供人阅读。"""

    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


class GameStatus(Enum):
    ONGOING = "ongoing"
    WIN = "win"
    DRAW = "draw"

class Chessboard:

    
    def __init__(self, row, col):
        for size in (row, col):
            if (isinstance(size, (bool, np.bool_))
                    or not isinstance(size, (int, np.integer)) or size <= 0):
                raise ValueError("棋盘行数和列数必须为正整数")
        self.row = int(row)
        self.col = int(col)
        self.b = -1 #黑方用-1代表
        self.w = 1 #白方用1代表
        self.board = np.zeros((self.row, self.col), dtype=np.int8)
        self.reset()

    "清空棋盘函数"
    def reset(self):
        """清空棋盘并恢复全部对局状态。"""
        self.board.fill(0)
        self.step_nums = 0
        self.terminated = False
        self.winner = 0
        self.status = GameStatus.ONGOING
        return self.observation()

    @property
    def current_player(self):
        """按步数轮换；终局后该值不再代表可行动的玩家。"""
        return self.b if self.step_nums % 2 == 0 else self.w

    def observation(self):
        """返回独立副本，外部修改观测不会改变棋盘。"""
        return {
            "board": self.board.copy(),
            "current_player": None if self.terminated else self.current_player,
            "terminated": self.terminated,
            "winner": self.winner,
            "status": self.status,
        }
    "观测的网络输入编码函数，输出己方棋盘布局和对手棋盘布局"
    @staticmethod
    def encoder(obs:dict) -> np.ndarray:
        board = obs["board"]
        if not isinstance(board, np.ndarray) or board.ndim != 2:
            raise ValueError("observation['board'] 必须是二维 ndarray")
        player = obs["current_player"]
        if player is None:
        # 终局：胜负已定，由调用方处理，不进入策略网络
            raise ValueError("终局观测没有 current_player,不应输入网络")

        if isinstance(player, (bool, np.bool_)) or player not in (-1, 1):
            raise ValueError(f"非法的 current_player: {player!r}")
        row , col = board.shape
        out = np.empty((2,row,col), dtype= np.float32)
        out[0] = (board == player)
        out[1] = (board == -player)
        return out
    def clone(self):
        """复制对局供搜索使用，模拟落子不会改变真实棋盘。"""
        other = Chessboard(self.row, self.col)
        other.board[:] = self.board
        other.step_nums = self.step_nums
        other.terminated = self.terminated
        other.winner = self.winner
        other.status = self.status
        return other

    "返回所有合法的落子位置（稀疏）"
    def legal_actions(self):
        if self.terminated:
            return []
        return np.flatnonzero(self.board == 0).tolist()
    "全局合法位置掩码数组（稠密）"
    def action_mask(self):
        if self.terminated:
            return np.zeros(self.row * self.col, dtype=bool)   # 终局：全 False
        return (self.board.reshape(-1) == 0)
    
    def check_win(self, row, col, player):
        """
        判断在 (row, col) 落子后,player 是否获胜。
        player: 黑方 -1,白方 1。
        """
        if not (0 <= row < self.row and 0 <= col < self.col):
            return False
        if player not in (self.b, self.w) or self.board[row, col] != player:
            return False
        # 定义四个方向轴：(行变化量, 列变化量)
        # 分别是：水平、垂直、主对角线(\)、副对角线(/)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # 遍历四个方向
        for dr, dc in directions:
            count = 1  # 包含刚刚落下的这颗子
            
            # 1. 正向延伸 (比如向右、向下、向右下、向左下)
            r, c = row + dr, col + dc
            while 0 <= r < self.row and 0 <= c < self.col and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
                
            # 2. 反向延伸 (比如向左、向上、向左上、向右上)
            r, c = row - dr, col - dc
            while 0 <= r < self.row and 0 <= c < self.col and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
                
            # 如果某一个方向上连成 5 个或以上，就赢了
            if count >= 5:
                return True
                
        return False # 四个方向都没凑齐 5 个

    def step(self, action):
        """返回 (观测, 是否终局, 辅助信息)，训练结果标签由赢家计算。

        所有动作检查均先于状态修改；非法动作抛 InvalidActionError。
        这是自定义接口，不是 Gymnasium 的五项返回接口。
        """
        if self.terminated:
            raise InvalidActionError(StepErrorCode.GAME_OVER, "游戏已结束，请先 reset()")
        if (isinstance(action, (bool, np.bool_))
                or not isinstance(action, (int, np.integer))):
            raise InvalidActionError(StepErrorCode.INVALID_TYPE, "action 必须为整数")
        action = int(action)
        if not 0 <= action < self.row * self.col:
            raise InvalidActionError(StepErrorCode.OUT_OF_BOUNDS, "action 超出棋盘范围")
        row, col = divmod(action, self.col)
        if self.board[row, col] != 0:
            raise InvalidActionError(StepErrorCode.OCCUPIED, "该位置已有棋子")

        acting_player = self.current_player
        self.board[row, col] = acting_player
        self.step_nums += 1

        
        if self.check_win(row, col, acting_player):
            self.winner = acting_player
            self.status = GameStatus.WIN
            self.terminated = True
            
        elif self.step_nums == self.row * self.col:
            self.status = GameStatus.DRAW
            self.terminated = True

        info = {
            "acting_player": acting_player,
            "last_move": (row, col),
            "winner": self.winner,
            "status": self.status,
        }
        return self.observation(), self.terminated, info
