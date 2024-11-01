import click
import pathlib

@click.command()
@click.argument('checkpoint_file', nargs=1, type=click.Path(exists=False, dir_okay=False, writable=True, path_type=pathlib.Path))
def cli(checkpoint_file):
    """
    A script creating a randomly initialized ByteLlama model checkpoint.
    """
    import torch
    from torch import nn

    from bytellama.builder import byte_llama

    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    model = byte_llama()
    model = model.apply(_init_weights)

    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    print(f'Writing model state_dict to {checkpoint_file}')
    with open(checkpoint_file, 'wb') as fp:
        torch.save(model.state_dict(), fp)
