from torch import nn
import torch


def create_mapper(encoder_dim, decoder_dim, prefix_size):
    modules = [nn.Linear(encoder_dim, (prefix_size * decoder_dim) // 2),
               nn.GELU(),
               nn.Linear((prefix_size * decoder_dim) // 2, prefix_size * decoder_dim)]
    return nn.Sequential(*modules)


class CapinchoMapper(nn.Module):
    def __init__(self, input_size, token_length, output_n):
        super(CapinchoMapper, self).__init__()
        self._tied_weights_keys = None
        self.model = nn.Sequential(
            nn.Linear(input_size, (token_length * output_n) // 2),
            nn.LeakyReLU(),
            nn.Linear((token_length * output_n) // 2, (token_length * output_n)),
            nn.LeakyReLU())

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = create_mapper(768, 512, 10)
    x = torch.randn(4, 768)
    x = model.forward(x)
    print(x.shape)