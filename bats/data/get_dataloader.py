import numpy as np
from typing import Any
from pathlib import Path
from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, TensorDataset

from bats.tools import StandardScaler
from bats.tools import print_log

FEATURE_MAIN = 0
FEATURE_TOD = 1
FEATURE_DOW = 2


def select_dataloader(task: str) -> Callable:
    task_upper = task.upper()

    if task_upper == 'LTSF':
        return build_LTSF_dataloader
    else:
        raise ValueError(f'{task} dataloader has not been implemented yet')


def _build_features(tod: bool, dow: bool) -> list[int]:
    features = [FEATURE_MAIN]
    if tod:
        features.append(FEATURE_TOD)
    if dow:
        features.append(FEATURE_DOW)
    return features


def _slice_data(
    data: np.ndarray, indices: np.ndarray, x_features: list[int], y_features: list[int]
) -> tuple[np.ndarray, np.ndarray]:

    num_samples = len(indices)
    x_len = indices[0, 1] - indices[0, 0]
    y_len = indices[0, 2] - indices[0, 1]
    num_nodes = data.shape[1]

    # 数据格式: (timesteps, nodes, features)
    # 输出格式: (num_samples, seq_len, num_nodes, num_features)
    x_data = np.empty(
        (num_samples, x_len, num_nodes, len(x_features)), dtype=data.dtype
    )
    y_data = np.empty(
        (num_samples, y_len, num_nodes, len(y_features)), dtype=data.dtype
    )

    for i, (start, mid, end) in enumerate(indices):
        x_data[i] = data[start:mid, :, x_features]  # (x_len, num_nodes, num_features)
        y_data[i] = data[mid:end, :, y_features]  # (y_len, num_nodes, num_features)

    return x_data, y_data


def _create_dataloaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    trainset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    valset = TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
    testset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

    persistent = num_workers > 0

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader


def _log_dataset_shapes(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    log: Any = None,
) -> None:
    print_log(f'Trainset:\tx-{x_train.shape}\ty-{y_train.shape}', log=log)
    print_log(f'Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}', log=log)
    print_log(f'Testset:\tx-{x_test.shape}\ty-{y_test.shape}', log=log)


def build_LTSF_dataloader(
    data_dir: str,
    batch_size: int = 32,
    in_steps: int = 96,
    out_steps: int = 96,
    x_tod: bool = False,
    x_dow: bool = False,
    y_tod: bool = False,
    y_dow: bool = False,
    log: Any = None,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:

    data_path = Path(data_dir)
    data = np.load(data_path / 'processed_data.npz')['data']
    index = np.load(data_path / f'index_in{in_steps}_out{out_steps}.npz')

    x_features = _build_features(x_tod, x_dow)
    y_features = _build_features(y_tod, y_dow)

    train_index = index['train']  # [num_samples, 3]
    val_index = index['val']
    test_index = index['test']

    len_train = train_index[-1][1]
    scaler = StandardScaler(
        mean=data[:len_train, :, FEATURE_MAIN].mean(axis=0),
        std=data[:len_train, :, FEATURE_MAIN].std(axis=0),
    )

    data[..., FEATURE_MAIN] = scaler.transform(data[..., FEATURE_MAIN])

    x_train, y_train = _slice_data(data, train_index, x_features, y_features)
    x_val, y_val = _slice_data(data, val_index, x_features, y_features)
    x_test, y_test = _slice_data(data, test_index, x_features, y_features)

    _log_dataset_shapes(x_train, y_train, x_val, y_val, x_test, y_test, log)
    print_log('INFO: Using scaled X and Y for LTSF task', log=log)

    train_loader, val_loader, test_loader = _create_dataloaders(
        x_train, y_train, x_val, y_val, x_test, y_test, batch_size
    )

    return train_loader, val_loader, test_loader, scaler
