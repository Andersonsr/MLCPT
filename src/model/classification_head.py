import torch
from torch import nn

# removed 'No Finding'
mimic_classifier_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                         'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
                         'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Findings']


class LinearClassifier(nn.Module):
    def __init__(self, input_size, output_classes):
        super(LinearClassifier, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_size, output_classes))

    def forward(self, x):
        return self.mlp(x)


class MultiClassifier(nn.Module):
    def __init__(self, classifiers_list, input_size, output_classes):
        super(MultiClassifier, self).__init__()
        for name in classifiers_list:
            self.add_module(name, LinearClassifier(input_size, output_classes))

    def forward(self, x):
        y = {}
        if len(x.shape) == 2:
            for name, module in self.named_children():
                # print(type(module))
                y[name] = module(x)
            return y

        if len(x.shape) == 3:
            # mapper output
            for i, child in enumerate(self.named_children()):
                name, module = child
                # print(x[:, i, :].shape)
                y[name] = module(x[:, i, :])
            return y


class Attention(nn.Module):
    def __init__(self, num_queries, embedding_dim):
        super(Attention, self).__init__()
        self.queries = torch.nn.Parameter(torch.rand(1, num_queries, embedding_dim))
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8, add_bias_kv=True, batch_first=True)

    def forward(self, x):
        batch_size, num_patches, embedding_dim = x.shape
        queries = self.queries.expand(batch_size, -1, -1)
        attn_output, attn_output_weights = self.attention(queries, x, x)
        return {'attn_output': attn_output, 'attn_output_weights': attn_output_weights}


if __name__ == '__main__':
    model = MultiClassifier(mimic_classifier_list, 896, 4)
    input = torch.rand((16, 14, 896))
    output = model(input)
    print(output)

