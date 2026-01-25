from collections.abc import Callable
import json
import numpy as np
import torch
import torch.nn as nn


def select_loss(loss: str) -> Callable:

    loss_upper = loss.upper()
    loss_mapping = {'MAE': nn.L1Loss, 'MSE': nn.MSELoss, 'HUBER': nn.HuberLoss}

    if loss_upper not in loss_mapping:
        raise ValueError(
            f'Invalid loss: {loss}. Supported: {list(loss_mapping.keys())}'
        )

    return loss_mapping[loss_upper]


def _compute_mask(y_true: np.ndarray, null_val: float | int = 0) -> np.ndarray:
    if np.isnan(null_val):
        mask = ~np.isnan(y_true)
    else:
        mask = np.not_equal(y_true, null_val)

    mask = mask.astype('float32')
    mask_mean = np.mean(mask)

    # 避免除以零
    if mask_mean > 0:
        mask /= mask_mean

    return mask


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError('y_true and y_pred must be numpy arrays')

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f'Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}'
        )


def mse(y_true: np.ndarray, y_pred: np.ndarray, null_val: float | int = 0) -> float:

    _validate_inputs(y_true, y_pred)

    with np.errstate(divide='ignore', invalid='ignore'):
        mask = _compute_mask(y_true, null_val)
        mse_values = np.square(y_pred - y_true)
        mse_values = np.nan_to_num(mse_values * mask)
        return float(np.mean(mse_values))


def mae(y_true: np.ndarray, y_pred: np.ndarray, null_val: float | int = 0) -> float:

    _validate_inputs(y_true, y_pred)

    with np.errstate(divide='ignore', invalid='ignore'):
        mask = _compute_mask(y_true, null_val)
        mae_values = np.abs(y_pred - y_true)
        mae_values = np.nan_to_num(mae_values * mask)
        return float(np.mean(mae_values))


def rmse(y_true: np.ndarray, y_pred: np.ndarray, null_val: float | int = 0) -> float:

    _validate_inputs(y_true, y_pred)

    with np.errstate(divide='ignore', invalid='ignore'):
        mask = _compute_mask(y_true, null_val)
        rmse_values = np.square(y_pred - y_true)
        rmse_values = np.nan_to_num(rmse_values * mask)
        return float(np.sqrt(np.mean(rmse_values)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, null_val: float | int = 0) -> float:

    _validate_inputs(y_true, y_pred)

    with np.errstate(divide='ignore', invalid='ignore'):
        mask = _compute_mask(y_true, null_val)

        # 先应用掩码，避免除以零
        y_true_masked = np.where(mask > 0, y_true, 1)
        y_pred_masked = np.where(mask > 0, y_pred, 0)

        mape_values = np.abs(
            np.divide((y_pred_masked - y_true_masked).astype('float32'), y_true_masked)
        )
        mape_values = np.nan_to_num(mask * mape_values)

        return float(np.mean(mape_values) * 100)


def compute_mse_mae(
    y_true: np.ndarray, y_pred: np.ndarray, null_val: float | int = 0
) -> tuple[float, float]:
    return mse(y_true, y_pred, null_val), mae(y_true, y_pred, null_val)


def compute_rmse_mae_mape(
    y_true: np.ndarray, y_pred: np.ndarray, null_val: float | int = 0
) -> tuple[float, float, float]:
    return (
        rmse(y_true, y_pred, null_val),
        mae(y_true, y_pred, null_val),
        mape(y_true, y_pred, null_val),
    )


class StandardScaler:
    def __init__(self, mean: float | None = None, std: float | None = None):

        self.mean = mean
        self.std = std

    def _validate_data(self, data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise ValueError('data must be a numpy array')

        if data.size == 0:
            raise ValueError('data cannot be empty')

    def fit_transform(self, data: np.ndarray) -> np.ndarray:

        self._validate_data(data)

        self.mean = data.mean()
        self.std = data.std()

        if self.std == 0:
            raise ValueError('Standard deviation is zero, cannot normalize data')

        return (data - self.mean) / self.std

    def transform(self, data: np.ndarray) -> np.ndarray:

        self._validate_data(data)

        if self.mean is None or self.std is None:
            raise ValueError('Scaler has not been fitted. Call fit_transform first.')

        if self.std == 0:
            raise ValueError('Standard deviation is zero, cannot normalize data')

        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:

        self._validate_data(data)

        if self.mean is None or self.std is None:
            raise ValueError('Scaler has not been fitted. Call fit_transform first.')

        return (data * self.std) + self.mean


def print_log(*values: object, log: str | None = None, end: str = '\n') -> None:

    print(*values, end=end)

    # 写入日志文件
    if log:
        try:
            with open(log, 'a', encoding='utf-8') as log_file:
                print(*values, file=log_file, end=end)
                log_file.flush()
        except IOError as e:
            print(f'Warning: Failed to write to log file {log}: {e}')


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: object) -> object:

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            # 将 numpy 数组转换为列表
            return obj.tolist()
        elif isinstance(obj, torch.device):
            return str(obj)
        else:
            return super().default(obj)
