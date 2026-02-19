import math
import torch
import torch.nn as nn


class FourierFilter(nn.Module):
    """
    Fourier Filter: to time-variant and time-invariant term
    """

    def __init__(self, mask_spectrum):
        super().__init__()
        self.mask_spectrum = mask_spectrum

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        mask = torch.ones_like(xf)
        mask[:, self.mask_spectrum, :] = 0
        x_var = torch.fft.irfft(xf * mask, dim=1)
        x_inv = x - x_var

        return x_var, x_inv
