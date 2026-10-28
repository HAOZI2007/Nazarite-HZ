"""棋盘规则检查与随机压测。

项目根目录运行：python -m tools.test.test_player --games 5000 --seed 42
也可直接运行本文件；仅运行固定用例：python tools/test/test_player.py --rules-only
"""

import argparse
from functools import lru_cache
from pathlib import Path
import random
import sys
import unittest

import numpy as np

# 直接运行脚本时，Python 默认只把 tools/test 放进模块搜索路径。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scene.board import Chessboard, InvalidActionError, StepErrorCode, GameStatus


def require(condition, message):
    """即使使用 python -O，也不跳过随机压测的检查。"""
    if not condition:
        raise AssertionError(message)


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} 必须为正整数")
    return int(value)


@lru_cache(maxsize=32)
def winning_windows(rows, cols):
    """枚举所有长度为五的窗口，独立于棋盘的双向计数实现。"""
    windows = []
    for row in range(rows):
        for col in range(cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if 0 <= row + 4 * dr < rows and 0 <= col + 4 * dc < cols:
                    windows.append([(row + k * dr) * cols + col + k * dc for k in range(5)])
    return np.asarray(windows, dtype=np.intp).reshape(-1, 5)


class GameVerifier:
    """维护独立棋面，每一步验证棋盘及其返回值。"""

    def __init__(self, board):
        self.board = board
        self.expected = np.zeros((board.row, board.col), dtype=np.int8)
        self.history = []
        self.status = GameStatus.ONGOING
        self.winner = 0
        self.check_state(board.observation())

    def legal_actions(self):
        if self.status != GameStatus.ONGOING:
            return []
        return np.flatnonzero(self.expected == 0).tolist()

    def check_state(self, obs):
        board = self.board
        ended = self.status != GameStatus.ONGOING
        player = None if ended else (-1 if len(self.history) % 2 == 0 else 1)
        require(np.array_equal(board.board, self.expected), "实际棋面与预期不一致")
        require(board.step_nums == len(self.history), "落子计数错误")
        require(board.status == self.status, "胜负状态错误")
        require(board.winner == self.winner, "获胜方错误")
        require(board.terminated == ended, "棋盘终局标志错误")
        require(board.legal_actions() == self.legal_actions(), "合法动作有遗漏、重复或包含已占用位置")
        if not ended:
            require(board.current_player == player, "玩家未正确轮换")
        require(np.array_equal(obs['board'], self.expected), "返回的观测棋面错误")
        for key, value in dict(current_player=player, terminated=ended,
                               winner=self.winner, status=self.status).items():
            require(obs[key] == value, f"观测字段 {key} 错误")
        require(not np.shares_memory(obs['board'], board.board), "观测棋面与内部棋盘共享内存")

    def step(self, action):
        require(action in self.legal_actions(), f"测试动作不合法：{action}")
        player = -1 if len(self.history) % 2 == 0 else 1
        row, col = divmod(int(action), self.board.col)
        self.history.append(int(action))
        self.expected[row, col] = player
        windows = winning_windows(*self.expected.shape)
        won = bool(np.any(np.all(self.expected.ravel()[windows] == player, axis=1)))
        if won:
            self.status, self.winner = GameStatus.WIN, player
        elif len(self.history) == self.expected.size:
            self.status = GameStatus.DRAW

        obs, terminated, info = self.board.step(action)
        self.check_state(obs)
        require(terminated == (self.status != GameStatus.ONGOING), "step 返回的终局标志错误")
        for key, value in dict(acting_player=player, last_move=(row, col),
                               winner=self.winner, status=self.status).items():
            require(info[key] == value, f"info 字段 {key} 错误")


def run_random_game(board_size=8, verbose=False, *, seed=None):
    """返回 (终局状态, 获胜方, 步数)，同一 seed 可复现落子序列。"""
    size = positive_integer(board_size, 'board_size')
    rng = random.Random(seed)
    board = Chessboard(size, size)
    verifier = GameVerifier(board)
    try:
        # 有限循环保证即使终局逻辑出错，测试也不会无限落子。
        for _ in range(size * size):
            legal = verifier.legal_actions()
            require(bool(legal), "游戏未结束但没有合法落点")
            action = rng.choice(legal)
            verifier.step(action)
            if verbose:
                print(f"第 {len(verifier.history)} 步，动作 {action}")
            if verifier.status != GameStatus.ONGOING:
                break
        require(board.terminated, "棋盘填满后仍未结束")
    except Exception:
        print(f"\n失败对局：board_size={size}, seed={seed}", file=sys.stderr)
        print(f"动作序列（含失败动作）：{verifier.history}", file=sys.stderr)
        print(board.board, file=sys.stderr)
        raise
    return board.status, board.winner, len(verifier.history)


def stress_test(num_games=10000, board_size=8, *, seed=None):
    """每局派生独立种子；失败时保留 traceback 并以非零状态退出。"""
    num_games = positive_integer(num_games, 'num_games')
    board_size = positive_integer(board_size, 'board_size')
    if seed is None:
        seed = random.SystemRandom().getrandbits(64)
    rng = random.Random(seed)
    print(f"开始压测：{num_games} 局，{board_size}×{board_size}，seed={seed}")
    stats = {'black_wins': 0, 'white_wins': 0, 'draws': 0,
             'total_steps': 0, 'min_steps': board_size * board_size, 'max_steps': 0}
    for i in range(num_games):
        game_seed = rng.getrandbits(64)
        try:
            status, winner, steps = run_random_game(board_size, seed=game_seed)
        except Exception:
            print(f"第 {i + 1} 局失败，主 seed={seed}，本局 seed={game_seed}", file=sys.stderr)
            raise
        key = 'draws' if status == GameStatus.DRAW else ('black_wins' if winner == -1 else 'white_wins')
        stats[key] += 1
        stats['total_steps'] += steps
        stats['min_steps'] = min(stats['min_steps'], steps)
        stats['max_steps'] = max(stats['max_steps'], steps)
        if (i + 1) % 1000 == 0:
            print(f"已完成 {i + 1} 局")
    print("压测通过：每步棋面、合法动作、轮换、胜负及观测均通过校验。")
    for label, key in [('黑方胜', 'black_wins'), ('白方胜', 'white_wins'), ('平局', 'draws')]:
        print(f"{label}：{stats[key]} 局（{stats[key] / num_games:.2%}）")
    print(f"步数：平均 {stats['total_steps'] / num_games:.2f}，最短 {stats['min_steps']}，最长 {stats['max_steps']}")
    return dict(stats, num_games=num_games, board_size=board_size, seed=seed)


class BoardRuleTests(unittest.TestCase):
    """可由 unittest/pytest 收集，也在脚本压测前自动执行。"""

    def play_win(self, targets, winner=-1):
        board = Chessboard(7, 9)  # 非正方形棋盘同时验证动作索引换算。
        game = GameVerifier(board)
        fillers = [(6, col) for col in (0, 2, 4, 6, 1, 8)]
        for i, target in enumerate(targets):
            pair = [target, fillers[i]] if winner == -1 else [fillers[i], target]
            for row, col in pair:
                game.step(row * board.col + col)
                if game.status != GameStatus.ONGOING:
                    break
        self.assertEqual(game.status, GameStatus.WIN)
        self.assertEqual(game.winner, winner)
        return game

    def test_four_directions_both_players(self):
        for dr, dc, start_col in [(0, 1, 0), (1, 0, 0), (1, 1, 0), (1, -1, 4)]:
            for player in (-1, 1):
                with self.subTest(direction=(dr, dc), player=player):
                    self.play_win([(i * dr, start_col + i * dc) for i in range(5)], player)

    def test_bridge_overline(self):
        # 最后一子补在中间，一次连接成六子，仍应获胜。
        self.play_win([(0, col) for col in (0, 1, 3, 4, 5, 2)])

    def test_draw_and_last_cell_win(self):
        for rows, cols in [(1, 1), (2, 3), (4, 4)]:
            with self.subTest(shape=(rows, cols)):
                game = GameVerifier(Chessboard(rows, cols))
                for action in range(rows * cols):
                    game.step(action)
                self.assertEqual(game.status, GameStatus.DRAW)
                self.assertEqual(game.winner, 0)
        game = GameVerifier(Chessboard(1, 9))
        for action in (0, 1, 2, 3, 6, 5, 8, 7, 4):
            game.step(action)
        self.assertEqual(game.status, GameStatus.DRAW)
        # 1×9 满盘的最后一步形成五连，胜利应优先于平局。
        game = GameVerifier(Chessboard(1, 9))
        for action in (0, 5, 1, 6, 3, 7, 4, 8, 2):
            game.step(action)
        self.assertEqual(game.status, GameStatus.WIN)

    def assert_rejected(self, board, action, code):
        before = board.observation()
        steps = board.step_nums
        legal = board.legal_actions()
        with self.assertRaises(InvalidActionError) as caught:
            board.step(action)
        self.assertEqual(caught.exception.code, code)
        after = board.observation()
        np.testing.assert_array_equal(before.pop('board'), after.pop('board'))
        self.assertEqual(before, after)
        self.assertEqual(board.step_nums, steps)
        self.assertEqual(board.legal_actions(), legal)

    def test_invalid_actions_do_not_mutate_state(self):
        board = Chessboard(3, 4)
        for action in (True, np.bool_(False), 1.0, '1', None, [1], np.array([1])):
            with self.subTest(action=repr(action)):
                self.assert_rejected(board, action, StepErrorCode.INVALID_TYPE)
        for action in (-1, 12, np.int64(100)):
            self.assert_rejected(board, action, StepErrorCode.OUT_OF_BOUNDS)
        board.step(np.int64(0))
        self.assert_rejected(board, 0, StepErrorCode.OCCUPIED)
        ended = Chessboard(1, 1)
        ended.step(0)
        self.assert_rejected(ended, 0, StepErrorCode.GAME_OVER)
        won = self.play_win([(0, col) for col in range(5)]).board
        self.assert_rejected(won, 20, StepErrorCode.GAME_OVER)

    def test_reset_and_observation_isolation(self):
        for state in ('ongoing', 'draw', 'win'):
            with self.subTest(state=state):
                if state == 'win':
                    board = self.play_win([(0, col) for col in range(5)]).board
                else:
                    board = Chessboard(1, 1) if state == 'draw' else Chessboard(3, 4)
                    board.step(0)
                obs = board.observation()
                obs['board'].fill(99)
                self.assertNotIn(99, board.board)
                reset_obs = board.reset()
                game = GameVerifier(board)
                game.check_state(reset_obs)
                game.step(0)
                self.assertTrue(np.all(reset_obs['board'] == 0))

    def test_invalid_dimensions_and_counts(self):
        for bad in (0, -1, True, np.bool_(True), 2.5, '8', None):
            with self.subTest(value=bad):
                for shape in ((bad, 8), (8, bad)):
                    with self.assertRaises(ValueError):
                        Chessboard(*shape)
                with self.assertRaises(ValueError):
                    stress_test(num_games=bad)
                with self.assertRaises(ValueError):
                    run_random_game(board_size=bad)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=5000, help='随机对局数（默认 5000）')
    parser.add_argument('--size', type=int, default=8, help='棋盘边长（默认 8）')
    parser.add_argument('--seed', type=int, help='随机种子，省略时自动生成并打印')
    parser.add_argument('--rules-only', action='store_true', help='仅执行固定规则用例')
    args = parser.parse_args()
    for name in ('games', 'size'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} 必须为正整数')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BoardRuleTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    if not args.rules_only:
        stress_test(args.games, args.size, seed=args.seed)
    return 0


if __name__ == '__main__':
    sys.exit(main())
