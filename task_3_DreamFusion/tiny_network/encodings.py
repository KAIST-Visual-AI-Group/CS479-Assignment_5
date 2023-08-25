from abc import abstractmethod
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

from .base import BaseModule
from .math import expected_sin
from .utils import print_tcnn_warning

try:
    import tinycudann as tcnn

    TCNN_EXISTS = True
except ModuleNotFoundError:
    TCNN_EXISTS = False


class Encoding(BaseModule):
    def __init__(self, in_dim):
        super().__init__(in_dim=in_dim)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError


class Identity(Encoding):
    def get_out_dim(self):
        return self.in_dim

    def forward(self, x):
        return x


class NeRFEncoding(Encoding):
    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = False,
        implementation: Literal["tcnn", "torch"] = "torch",
    ):
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input

        self.tcnn_encoding = None
        if implementation == "tcnn" and not TCNN_EXISTS:
            print_tcnn_warning("NeRFEncoding")
        elif implementation == "tcnn":
            encoding_config = {"otype": "Frequency", "n_frequencies": num_frequencies}
            assert min_freq_exp == 0, "tcnn only supports min_freq_exp = 0"
            assert (
                max_freq_exp == num_frequencies - 1
            ), "tcnn only supports max_freq_exp = num_frequencies - 1"
            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=in_dim,
                encoidng_config=encoding_config,
            )

    def get_out_dim(self):
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def pytorch_fwd(
        self,
        x: Float[torch.Tensor, "B N"],
        covs: Optional[Float[torch.Tensor, "B N N"]] = None,
    ):
        scaled_x = 2 * torch.pi * x
        freqs = 2 ** torch.linspace(
            self.min_freq, self.max_freq, self.num_frequencies
        ).to(x.device)
        scaled_inputs = scaled_x = scaled_x[..., None] * freqs
        scaled_inputs = scaled_inputs.view(*scaled_x.shape[:-2], -1)

        if covs is None:
            encoded_inputs = torch.sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1)
            )
        else:
            input_var = (
                torch.diagnoal(covs, dim1=-2, dim2=-1)[..., :, None]
                * freqs[None, :] ** 2
            )
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1),
                torch.cat([input_var] * 2, dim=-1),
            )

        if self.incpude_input:
            encoded_inputs = torch.cat([encoded_inputs, x], dim=-1)
        return encoded_inputs

    def forward(
        self, x: Float[torch.Tensor, "B N"],
        covs: Optional[Float[torch.Tensor, "B N N"]] = None
    ):
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(x)
        return self.pytorch_fwd(x, covs)


class HashEncoding(Encoding):
    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
        implementation: Literal["tcnn", "torch"] = "torch",
        interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = None,
    ):
        super().__init__(in_dim=3)
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size =  2**log2_hashmap_size

        levels = torch.arange(num_levels)
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1))
        self.scalings = torch.floor(min_res * growth_factor**levels)

        self.hash_offset = levels * self.hash_table_size
        self.hash_table = torch.rand(size=(self.hash_table_size * num_levels, features_per_level)) * 2 - 1
        self.hash_table *= hash_init_scale
        self.hash_table = nn.Parameter(self.hash_table)

        self.tcnn_encoding = None
        if implementation == "tcnn" and not TCNN_EXISTS:
            print_tcnn_warning("HashEncoding")
        elif implementation == "tcnn":
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": self.num_levels,
                "n_features_per_level": self.features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": min_res,
                "per_level_scale": growth_factor,
            }
            if interpolation is not None:
                encoding_config["interpolation"] = interpolation

            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )

        if self.tcnn_encoding is None:
            assert (
                interpolation is None or interpolation == "Linear"
            ), f"interpolation '{interpolation}' is not supported for torch encoding backend"

    def get_out_dim(self):
        return self.num_levels * self.features_per_level

    def hash_fn(self, in_tensor: Int[torch.Tensor, "B num_levels 3"]):
        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861]).to(in_tensor.device)
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += self.hash_offset.to(x.device)
        return x

    def pytorch_fwd(self, in_tensor: Float[torch.Tensor, "*bs input_dim"]) -> Float[torch.Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device)  # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]

    def forward(self, in_tensor: Float[torch.Tensor, "*bs input_dim"]) -> Float[torch.Tensor, "*bs output_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)

