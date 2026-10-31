"""uv run train TASK；也支持 python -m train_manager.scripts.train TASK。"""
import argparse

from train_manager.checkpoint import load_checkpoint
from train_manager.config import TrainConfig
from train_manager.registry import get_task, list_tasks


def main(argv=None):
    import train_manager.tasks  # 显式加载内置任务注册

    parser = argparse.ArgumentParser(description='MCTS 自博弈五子棋训练')
    parser.add_argument('task', nargs='?', help='任务名字')
    parser.add_argument('--list', action='store_true', help='列出内置任务')
    parser.add_argument('--resume', help='恢复可信的本项目 checkpoint')
    parser.add_argument('--iterations', type=int, help='目标总轮数，不是额外增加的轮数')
    parser.add_argument('--device', help='auto / cpu / cuda / cuda:0')
    parser.add_argument('--output-dir', help='运行输出根目录')
    args = parser.parse_args(argv)
    if args.list:
        print('\n'.join(list_tasks()))
        return 0
    if args.task is None:
        parser.error('请指定任务名字，或使用 --list')
    try:
        spec = get_task(args.task)
        if args.resume:
            checkpoint = load_checkpoint(args.resume)
            if checkpoint['task_id'] != args.task:
                raise ValueError('断点任务名与命令行任务名不同')
            cfg = TrainConfig(**checkpoint['config'])
            del checkpoint  # 避免创建 Manager 时同时持有额外大份经验池
        else:
            cfg = spec.config_factory()
        for argument, field in (('iterations', 'iterations'), ('device', 'device'),
                                ('output_dir', 'output_dir')):
            value = getattr(args, argument)
            if value is not None:
                setattr(cfg, field, value)
        cfg.validate()
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    manager = spec.manager_cls(cfg, spec.model_factory, task_id=args.task)
    if args.resume:
        manager.restore(args.resume)
    manager.run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
