import time
import copy
import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary

from .BaseTrainer import BaseTrainer
from tools import print_log, compute_mse_mae


class LTSFTrainer(BaseTrainer):
    def __init__(
        self, cfg: dict, device: torch.device, scaler: Any, log: str | None = None
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.scaler = scaler
        self.log = log

        self.clip_grad = self.cfg['OPTIM'].get('clip_grad')

    def train_one_epoch(self, model, train_loader, optimizer, scheduler, criterion):
        model.train()

        batch_loss_list = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            out_batch = model(x_batch)

            loss = criterion(out_batch, y_batch)
            batch_loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
            optimizer.step()

        epoch_loss = np.mean(batch_loss_list)
        scheduler.step()

        return epoch_loss

    @torch.no_grad()
    def eval_model(self, model, val_loader, criterion):
        model.eval()

        batch_loss_list = []
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.float().to(self.device)
            y_batch = y_batch.float().to(self.device)

            out_batch = model(x_batch)

            loss = criterion(out_batch.detach().cpu(), y_batch.detach().cpu())
            batch_loss_list.append(loss.item())

        return np.mean(batch_loss_list)

    @torch.no_grad()
    def predict(self, model, loader):
        model.eval()

        y = []
        out = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.float().to(self.device)
            y_batch = y_batch.float().to(self.device)

            out_batch = model(x_batch)

            out_batch = out_batch.cpu().numpy()
            y_batch = y_batch.cpu().numpy()

            out.append(out_batch)
            y.append(y_batch)

        # (samples, out_steps, num_nodes, output_dim)
        out = np.vstack(out)
        y = np.vstack(y)

        return y, out

    def train_model(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        max_epochs=10,
        early_stop_patience=3,
        verbose=1,
        save=None,
    ):
        wait = 0
        min_val_loss = np.inf

        train_loss_list = []
        val_loss_list = []

        start = time.time()
        for epoch in range(max_epochs):
            train_loss = self.train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion
            )
            train_loss_list.append(train_loss)

            val_loss = self.eval_model(model, val_loader, criterion)
            val_loss_list.append(val_loss)

            if (epoch + 1) % verbose == 0:
                print_log(
                    datetime.datetime.now(),
                    'Epoch',
                    epoch + 1,
                    f' \tTrain Loss = {train_loss:.5f}',
                    f'Val Loss = {val_loss:.5f}',
                    log=self.log,
                )

            if val_loss < min_val_loss:
                wait = 0
                min_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= early_stop_patience:
                    break
        end = time.time()

        model.load_state_dict(best_state_dict)

        if save:
            torch.save(best_state_dict, save)

        train_mse, train_mae = compute_mse_mae(*self.predict(model, train_loader))
        val_mse, val_mae = compute_mse_mae(*self.predict(model, val_loader))

        out_str = f'Finish at epoch: {epoch + 1}\n'
        out_str += f'Best model at epoch {best_epoch + 1}:\n'

        out_str += f'Train Loss = {train_loss_list[best_epoch]:.5f}\n'
        out_str += f'Train MSE = {train_mse:.5f}, MAE = {train_mae:.5f}\n'
        out_str += f'Val Loss = {val_loss_list[best_epoch]:.5f}\n'
        out_str += f'Val MSE = {val_mse:.5f}, MAE = {val_mae:.5f}'

        print_log(out_str, log=self.log)
        print_log(
            f'Training time per epoch: {(end - start) / epoch:.3f} s', log=self.log
        )

        return model

    @torch.no_grad()
    def test_model(self, model, test_loader):
        model.eval()

        print_log('--------- Test ---------', log=self.log)

        start = time.time()
        y_true, y_pred = self.predict(model, test_loader)
        end = time.time()

        out_steps = y_pred.shape[1]

        mse_all, mae_all = compute_mse_mae(y_true, y_pred)
        out_str = (
            f'All Steps (1-{out_steps}) MSE = {mse_all:.5f}, MAE = {mae_all:.5f}\n'
        )

        print_log(out_str, log=self.log, end='')
        print_log('Inference time: %.3f s' % (end - start), log=self.log)

    def model_summary(self, model, dataloader):
        x_shape = next(iter(dataloader))[0].shape

        return summary(model, x_shape, verbose=0, device=self.device)
