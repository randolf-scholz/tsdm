#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class Config:
    input_size: int
    output_size: int
    latent_size: int = None

    def __post_init__(self):
        if self.latent_size is None:
            self.latent_size = self.input_size


conf = Config(2, 3)
bar: int = conf.latent_size
