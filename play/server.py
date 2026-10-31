"""不额外安装 Web 框架，复用训练网络和 MCTS 的本地对弈服务。"""

import argparse
import json
import logging
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from time import perf_counter

import torch

from algorithm.networks.LeNet.model import LeNet
from algorithm.search.mcts.mcts import MCTSPlanner
from scene.board import Chessboard
from train_manager.checkpoint import load_checkpoint


PROJECT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / 'static'


def integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} 必须为 {minimum}～{maximum} 的整数')
    return value


class PlayService:
    def __init__(self, model_dir, device='cpu'):
        self.model_dir = Path(model_dir).resolve()
        self.device = torch.device(device)
        self.lock = Lock()
        self.model = None
        self.info = None

    def models(self):
        files = sorted(self.model_dir.rglob('*.pt'),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        return [{'path': str(p), 'name': str(p.relative_to(self.model_dir))}
                for p in files]

    def load(self, path):
        if not isinstance(path, str) or not path.strip():
            raise ValueError('请选择模型或填写 checkpoint 路径')
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = PROJECT / target
        target = target.resolve()
        if not target.is_file() or target.suffix != '.pt':
            raise ValueError('找不到 .pt 模型文件，请检查路径')
        # 完整训练断点包含 NumPy/RNG 状态，沿用项目的可信断点加载器。
        with self.lock:
            data = load_checkpoint(target)
            expected = f'{LeNet.__module__}.{LeNet.__qualname__}'
            if data['model_class'] != expected:
                raise ValueError(f'当前支持 {expected}，此断点的网络结构不匹配')
            rows = integer(data['config']['rows'], '棋盘行数', 5, 25)
            cols = integer(data['config']['cols'], '棋盘列数', 5, 25)
            model = LeNet().to(self.device)
            model.load_state_dict(data['model'], strict=True)
            model.eval()
            with torch.inference_mode():
                logits, value = model(torch.zeros(1, 2, rows, cols, device=self.device))
                if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
                    raise ValueError('模型输出包含无效数值')
            info = {
                'path': str(target), 'name': target.parent.name,
                'task': data['task_id'], 'rows': rows, 'cols': cols,
                'iterations': data['counters']['completed_iterations'],
                'device': str(self.device),
            }
            self.model, self.info = model, info
            return info.copy()

    def _board(self, moves):
        if self.model is None:
            raise ValueError('请先加载模型')
        if not isinstance(moves, list) or len(moves) > self.info['rows'] * self.info['cols']:
            raise ValueError('无效的落子记录')
        board = Chessboard(self.info['rows'], self.info['cols'])
        for move in moves:
            board.step(move)
        return board

    @staticmethod
    def snapshot(board, moves):
        return {
            'board': board.board.tolist(), 'moves': moves,
            'current_player': None if board.terminated else board.current_player,
            'terminated': board.terminated, 'winner': board.winner,
            'status': board.status.value,
        }

    def position(self, moves):
        with self.lock:
            return self.snapshot(self._board(moves), moves)

    def move(self, moves, simulations):
        simulations = integer(simulations, '搜索次数', 1, 2000)
        with self.lock:
            board = self._board(moves)
            if board.terminated:
                raise ValueError('对局已经结束')
            start = perf_counter()
            planner = MCTSPlanner(self.model, simulation_limit=simulations,
                                  device=self.device)
            action = planner.plan(board)
            board.step(action)
            result = self.snapshot(board, [*moves, action])
            result.update(action=action, elapsed=round(perf_counter() - start, 3))
            return result


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/api/models':
                self.reply(200, {'models': service.models(), 'loaded': service.info})
                return
            pages = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}
            filename = pages.get(self.path)
            if filename is None:
                self.reply(404, {'error': '页面不存在'})
                return
            body = (STATIC / filename).read_bytes()
            self.send_response(200)
            mime = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            self.send_header('Content-Type', mime + '; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            # 浏览器只允许从此服务自己的页面发起修改请求。
            origin = self.headers.get('Origin')
            if origin and origin != f'http://{self.headers.get("Host")}':
                self.reply(403, {'error': '不允许跨站请求'})
                return
            if self.headers.get_content_type() != 'application/json':
                self.reply(415, {'error': '请求必须使用 JSON'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16384:
                    raise ValueError('请求大小无效')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError('请求必须为 JSON 对象')
                if self.path == '/api/load':
                    result = service.load(data.get('path'))
                elif self.path == '/api/position':
                    result = service.position(data.get('moves'))
                elif self.path == '/api/move':
                    result = service.move(data.get('moves'), data.get('simulations', 100))
                else:
                    self.reply(404, {'error': '接口不存在'})
                    return
                self.reply(200, result)
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                self.reply(400, {'error': str(exc)})
            except Exception:
                logging.exception('对弈服务异常')
                self.reply(500, {'error': '模型处理失败，请查看终端日志'})

    return Handler


def main():
    parser = argparse.ArgumentParser(description='五子棋本地人机对弈网页')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model-dir', type=Path, default=PROJECT / 'logs')
    parser.add_argument('--checkpoint', type=Path, help='启动时加载可信的训练断点')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    args = parser.parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    # 小棋盘的逐次前向使用一个 CPU 线程，避免线程调度开销。
    torch.set_num_threads(1)
    service = PlayService(args.model_dir, device)
    if args.checkpoint:
        service.load(str(args.checkpoint))
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(service))
    print(f'五子棋对弈页面：http://127.0.0.1:{server.server_port}', flush=True)
    print('仅在本机提供服务；按 Ctrl+C 停止。', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
