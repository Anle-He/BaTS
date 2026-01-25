import os
import json
import yaml
import random
import argparse
import datetime
import numpy as np
from typing import Any
from pathlib import Path
from rich.traceback import install

import torch

from models import select_model
from trainers import select_trainer
from data.get_dataloader import select_dataloader
from tools import print_log, select_loss, CustomJSONEncoder

install()

DATA_DIR = Path('data/datasets_zoo')
LOG_DIR = Path('logs')
CHECKPOINT_DIR = Path('checkpoints')

DEFAULT_MODEL = 'DLinear'
DEFAULT_TASK = 'LTSF'
DEFAULT_DATASET = 'PEMS08'
DEFAULT_SEED = 2024
DEFAULT_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, default=DEFAULT_MODEL)
    parser.add_argument('-t', '--task_name', type=str, default=DEFAULT_TASK)
    parser.add_argument('-d', '--dataset_name', type=str, default=DEFAULT_DATASET)
    parser.add_argument('-cfg', '--config_path', type=str, default=None)
    parser.add_argument('-sd', '--seed', type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict[str, Any]:
    config_file = Path(config_path)

    with config_file.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_log_file(model_name: str, task_name: str, dataset_name: str) -> Any:
    log_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = LOG_DIR / f'{model_name}_{dataset_name.upper()}'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = (
        log_dir
        / f'{model_name}-{task_name.upper()}-{dataset_name.upper()}-{log_time}.log'
    )

    log_file.write_text('')

    return log_file.open('a', encoding='utf-8')


def create_checkpoint_path(model_name: str, task_name: str, dataset_name: str) -> Path:

    log_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    checkpoint_dir = CHECKPOINT_DIR / f'{model_name}_{dataset_name.upper()}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return (
        checkpoint_dir
        / f'{model_name}-{task_name.upper()}-{dataset_name.upper()}-{log_time}.pt'
    )


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    cfg: dict[str, Any],
    train_loader_len: int,
) -> torch.optim.lr_scheduler._LRScheduler:

    scheduler_map = {
        'ExponentialLR': lambda: torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=cfg.get('lr_scheduler_gamma', 0.5)
        ),
        'OneCycleLR': lambda: torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            steps_per_epoch=train_loader_len,
            max_lr=cfg['OPTIM'].get('initial_lr'),
            epochs=cfg['GENERAL'].get('max_epochs'),
            pct_start=cfg['OPTIM'].get('lr_scheduler_pct_start', 0.3),
        ),
        'MultiStepLR': lambda: torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.get('milestones', []),
            gamma=cfg.get('lr_decay_rate', 0.1),
        ),
    }

    return scheduler_map[scheduler_type]()


def main() -> None:
    args = parse_args()

    set_random_seed(args.seed)

    device = torch.device(DEFAULT_DEVICE)

    cfg = load_config(args.config_path)

    model_arch = select_model(args.model_name)
    model = model_arch(**cfg['MODEL_PARAM']).to(device)

    log = create_log_file(args.model_name, args.task_name, args.dataset_name)

    print_log(f'Dataset used: {args.dataset_name.upper()}', log=log)
    data_path = DATA_DIR / args.dataset_name
    train_loader, val_loader, test_loader, scaler = select_dataloader(
        args.task_name.upper()
    )(
        data_path,
        batch_size=cfg['GENERAL'].get('batch_size', 32),
        in_steps=cfg['DATA'].get('in_steps', 96),
        out_steps=cfg['DATA'].get('out_steps', 12),
        x_tod=cfg['DATA'].get('x_time_of_day'),
        x_dow=cfg['DATA'].get('x_day_of_week'),
        y_tod=cfg['DATA'].get('y_time_of_day'),
        y_dow=cfg['DATA'].get('y_day_of_week'),
        log=log,
    )
    print_log(log=log)

    checkpoint_path = create_checkpoint_path(
        args.model_name, args.task_name, args.dataset_name
    )

    criterion = select_loss(cfg['OPTIM'].get('loss', 'MSE'))(
        **cfg['OPTIM'].get('loss_args', {})
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['OPTIM'].get('initial_lr', 1e-3)
    )

    lr_scheduler_type = cfg['OPTIM'].get('lr_scheduler_type', 'ExponentialLR')
    scheduler = create_lr_scheduler(
        optimizer, lr_scheduler_type, cfg['OPTIM'], len(train_loader)
    )

    trainer = select_trainer(cfg['GENERAL'].get('runner', 'LTSFTrainer'))(
        cfg, device=device, scaler=scaler, log=log
    )

    print_log('---------', args.model_name, '---------', log=log)
    print_log(f'Random Seed = {args.seed}', log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )

    print_log(f'Checkpoints saved at: {checkpoint_path}', log=log)
    print_log(log=log)
    model = trainer.train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        max_epochs=cfg['GENERAL'].get('max_epochs', 10),
        early_stop_patience=cfg['GENERAL'].get('early_stop_patience', 3),
        verbose=1,
        save=str(checkpoint_path),
    )

    trainer.test_model(model, test_loader)

    log.close()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
