dummy = ['cat', 'dog', 'airplane']

DatasetsClassDict = {'dummy': dummy}


def plot_curves(training, validation, output_name):
    import matplotlib.pyplot as plt
    plt.plot(training, label=f'training loss')

    if len(validation) > 0:
        scale = len(training) // len(validation)
        val_Step = [i*scale-1 for i in range(len(validation))]
        plt.plot(val_Step, validation, label=f'validation loss')
        plt.text(len(validation), validation[-1], f'{validation[-1]:.3}')

    plt.text(len(training), training[-1], f'{training[-1]:.3}')

    plt.title(f'training loss')
    plt.legend()
    plt.savefig(output_name)
    plt.clf()


def model_size(model):
    import torch
    size_model = 0
    for param in model.parameters():
        if param.is_floating_point():
            size_model += param.numel() * torch.finfo(param.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.dtype).bits
    return f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB"


def learnable_parameters(model):
    learnable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            learnable += param.numel()

    return f'total params: {total / 1e6:.2f}M,  learnable params: {learnable / 1e6:.2f}M'
