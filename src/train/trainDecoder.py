import argparse
import logging
import torch
from tqdm import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt
import json
import os
import sys
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
 checkpoint_wrapper,
 CheckpointImpl,
 apply_activation_checkpointing)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
# path trick
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from model.decoder import Decoder
from model.encoder import Encoder
from dataset.coco import COCOCaptions, COCOEmbeddings
from util import learnable_parameters


def train(decoder, train_loader, optimizer, rank, world_size, epoch, sampler):
    # print(f'batches {len(train_loader)}')
    local_rank = int(os.environ['RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)

    if sampler:
        sampler.set_epoch(epoch)

    if rank == 0:
        inner_pbar = tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )

    for batch in tqdm(train_loader, total=len(train_loader), desc='Epoch {}'.format(epoch)):
        optimizer.zero_grad()
        embeddings = batch['image_embeddings'].to(local_rank)

        optimizer.zero_grad()
        output = decoder(embeddings, batch['caption'])
        loss = output.loss
        loss.backward()
        optimizer.step()

        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch['caption'])
        if local_rank == 0:
            inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_loss = fsdp_loss[0] / fsdp_loss[1]

    if rank == 0:
        inner_pbar.close()
        print(
            f"Train Epoch: \t{epoch}, Loss: \t{train_loss:.4f}"
        )

    return train_loss


def validation(decoder, val_loader, rank, world_size, epoch):
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)

    if rank == 0:
        inner_pbar = tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )

    with torch.no_grad():
        for val_batch in val_loader:
            # validate using text embeddings in text only training
            embeddings = val_batch['image_embeddings']
            output = decoder(embeddings, val_batch['caption'])
            fsdp_loss[0] += output.loss.item()
            fsdp_loss[1] += len(val_batch['caption'])

            if local_rank == 0:
                inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]

    if rank == 0:
        inner_pbar.close()
        print(f"Val Epoch: \t{epoch} Validation Loss: {val_loss:.4f}")

    return val_loss


def train_main(args):
    os.environ["USE_LIBUV"] = '0'
    batch_size = getattr(args, 'batch_size')
    decoder_name = getattr(args, 'decoder_name')
    prefix_len = getattr(args, 'prefix_length')
    add_noise = getattr(args, 'add_noise')
    variance = getattr(args, 'noise_variance')
    lora = getattr(args, 'lora')
    lora_rank = getattr(args, 'lora_rank')
    lora_alpha = getattr(args, 'lora_alpha')
    lora_dropout = getattr(args, 'lora_dropout')
    lr = getattr(args, 'lr')
    epochs = getattr(args, 'epochs')
    root = getattr(args, 'save_dir')
    save_history = getattr(args, 'save_history')
    dataset = getattr(args, 'dataset')
    dataset_root = getattr(args, 'dataset_root')
    precomputed_embeddings = getattr(args, 'precomputed_embeddings')

    # FSDP stuff
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Dataset
    if dataset == 'coco':
        train_dataset = COCOEmbeddings(precomputed_embeddings)
        val_dataset = COCOEmbeddings(precomputed_embeddings.replace('train', 'val'))

    else:
        raise ValueError(f'{dataset} not supported')

    # dataloaders
    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': batch_size, 'sampler': train_sampler}
    val_kwargs = {'batch_size': batch_size, 'sampler': val_sampler}
    cuda_kwargs = {'num_workers': 2,
                   'pin_memory': True,
                   'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    val_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_kwargs)
    logging.debug('training dataset size: %d' % len(train_dataset))
    logging.debug('validation dataset size: %d' % len(val_dataset))

    dist.init_process_group("nccl")
    # dist.init_process_group("gloo")
    torch.cuda.set_device(local_rank)

    # model
    dim = train_dataset[0]['image_embeddings'].shape[1]
    decoder = Decoder(decoder_name,
                      prefix_length=prefix_len,
                      add_noise=add_noise,
                      variance=variance,
                      input_dimension=dim,
                      precision=torch.float32)

    if lora:
        # model was adapted before, load existing adapter to continue training
        if os.path.exists(os.path.join(decoder_name, 'adapter_config.json')):
            logging.debug('loaded existing adapter')
            decoder.model.enable_adapters()

        else:
            # create new adapter
            decoder.lora_model(lora_rank, lora_alpha, lora_dropout)
            logging.debug('created new adapter')

    # model to FSDP
    decoder = FSDP(decoder, auto_wrap_policy=size_based_auto_wrap_policy, device_id=torch.cuda.current_device())
    optim = AdamW(decoder.parameters(), lr=lr)

    logging.debug('DECODER SIZE {}'.format(learnable_parameters(decoder.model)))
    logging.debug('MAPPER SIZE {}'.format(learnable_parameters(decoder.mapper)))

    training_losses = []
    validation_losses = []

    # training loop
    for epoch in range(epochs):
        train_loss = train(decoder, train_loader, optim, rank, world_size, epoch, train_sampler)
        validation_loss = validation(decoder, val_loader, rank, world_size, epoch)
        training_losses.append(train_loss)
        validation_loss.append(validation_loss)

        if rank == 0:
            plt.plot(range(len(training_losses)), training_losses, label='training')
            plt.legend()
            plt.xlabel('step')
            plt.ylabel('loss')
            plt.title(f'training loss')

            plt.savefig(f'{root}/loss_plot.png')
            plt.clf()
            log = {'training_loss': training_losses, 'validation_loss': validation_losses}
            with open(f'{root}/loss_log.pkl', 'w') as f:
                json.dump(log, f)

            # epoch model
            cpu_state = decoder.state_dict()
            if save_history:
                torch.save(cpu_state, f'{root}/checkpoint_{epoch+1}.pt')
            else:
                torch.save(cpu_state, f'{root}/checkpoint.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lora', action='store_true', default=False)
    parser.add_argument('--lora_rank', type=int, default=16, help='lora rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='lora alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout parameter')
    parser.add_argument('--decoder_name', type=str, default="facebook/opt-350m", help='OPT model name')
    parser.add_argument('--encoder_name', type=str, default=None)
    parser.add_argument('--prefix_length', type=int, default=10, help='model prefix length')
    parser.add_argument('--add_noise', action='store_true', help='add noise to embeddings', default=False)
    parser.add_argument('--noise_variance', type=float, help='variance for noise injection', default=0.016)
    parser.add_argument('--save_history', action='store_true', help='save epoch history', default=False)
    parser.add_argument('--dataset', type=str, default='coco', help='dataset name',
                        choices=['coco', 'petro', 'petro-txt', 'cego', 'mimic'], )
    parser.add_argument('--save_dir', required=True, help='root dir for saving results')
    parser.add_argument('--logging_steps', type=int, default=None, help='log step')
    parser.add_argument('--debug', action='store_true', help='debug mode', default=False)
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp16'], help='data type')
    parser.add_argument('--frozen_encoder', action='store_true', help='freeze encoder', default=False)
    parser.add_argument('--dataset_root', type=str, help='path to dataset root folder')
    parser.add_argument('--precomputed_embeddings', type=str, help='path to precomputed embeddings train file', default=None)
    parser.add_argument('--no_eval', action='store_true', help='do not evaluate the model', default=False)

    args = parser.parse_args()
    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        logging.info(f'folders created: {args.save_dir}')

    precision = torch.float16 if args.dtype == 'fp16' else torch.float32
    logging.debug(f'precision: {precision}')

    # check for previously trained checkpoint
    cfg_path = os.path.join(args.decoder_name, 'adapter_config.json')
    if os.path.exists(cfg_path):
        logging.debug('decoder was adapted before locally')
        with open(cfg_path, 'rb') as f:
            cfg = json.load(f)
            args.rank = cfg['r']
            args.alpha = cfg['lora_alpha']
            logging.debug(f'decoder rank: {args.rank} alpha: {args.alpha}')

    # save parameters
    result_dict = args.__dict__
    result_dict['checkpoint_path'] = os.path.join(args.save_dir, 'checkpoint.pt')
    with open(f'{args.save_dir}/experiment.json', 'w') as f:
        json.dump(result_dict, f, indent=2)
        logging.info(f'experiment saved')

    train_main(args)
