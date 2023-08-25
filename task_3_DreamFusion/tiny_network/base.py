from abc import abstractmethod
from typing import Optional
import torch
import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(self, in_dim: Optional[int] = None, out_dim: Optional[int] = None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    @abstractmethod
    def build_nn_modules(self):
        raise NotImplementedError

    def set_in_dim(self, in_dim: int):
        if in_dim <= 0:
            raise ValueError("Input dim should be greater than zero")
        self.in_dim = in_dim
    
    def get_out_dim(self):
        if self.out_dim is None:
            raise ValueError("Output dim has not been set")
        return self.out_dim

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError
