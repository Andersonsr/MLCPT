from torch import nn


def create_mapper(encoder_dim, decoder_dim, prefix_size):
    modules = [nn.Linear(encoder_dim, (prefix_size * decoder_dim) // 2),
               nn.GELU(),
               nn.Linear((prefix_size * decoder_dim) // 2, decoder_dim * prefix_size),
               nn.Unflatten(-1, (prefix_size, decoder_dim))]
    return nn.Sequential(*modules)

