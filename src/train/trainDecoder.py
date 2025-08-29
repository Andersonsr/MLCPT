import argparse
import pickle
import random
import logging
import torch
from tqdm import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt
import json
import os
import sys
# path trick
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
print(path)
sys.path.append(path)
from model.decoder import Decoder
from model.encoder import Encoder
from dataset.coco import CocoCaptions
from util import learnable_parameters


def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('model device {}'.format(device))
    # data
    num_captions = 1
    dataset = getattr(args, 'dataset', None)

    if dataset == 'coco':
        val_data = CocoCaptions('E:/datasets/coco_2017/annotations/captions_val2017.json',
                                  'E:/datasets/coco_2017/',
                                  'val')
        train_data = CocoCaptions('E:/datasets/coco_2017/annotations/captions_train2017.json',
                                  'E:/datasets/coco_2017/',
                                  'train')

    logging.debug('training dataset size: %d' % len(train_data))
    logging.debug('validation dataset size: %d' % len(val_data))

    batch_size = getattr(args, 'batch_size')
    decoder_name = getattr(args, 'decoder_name')
    prefix_len = getattr(args, 'prefix_length')
    add_noise = getattr(args, 'add_noise')
    encoder_name = getattr(args, 'encoder_name')
    variance = getattr(args, 'noise_variance')
    lora = getattr(args, 'lora')
    rank = getattr(args, 'lora_rank')
    alpha = getattr(args, 'lora_alpha')
    dropout = getattr(args, 'lora_dropout')
    lr = getattr(args, 'lr')
    log_step = getattr(args, 'logging_steps')
    epochs = getattr(args, 'epochs')
    root = getattr(args, 'save_dir')
    save_history = getattr(args, 'save_history')
    dataset = getattr(args, 'dataset')
    frozen_encoder = getattr(args, 'frozen_encoder')

    train_loader = train_data.get_loader(batch_size=batch_size, shuffle=True)
    val_loader = val_data.get_loader(batch_size=batch_size, shuffle=True)

    # model
    encoder = Encoder(encoder_name, dataset, False)

    decoder = Decoder(decoder_name, device,
                      prefix_length=prefix_len,
                      add_noise=add_noise,
                      variance=variance,
                      input_dimension=encoder.dim,
                      precision=torch.float32)

    if lora:
        # model was adapted before, load existing adapter to continue training
        if os.path.exists(os.path.join(decoder_name, 'adapter_config.json')):
            logging.debug('loaded existing adapter')
            decoder.model.enable_adapters()

        else:
            # create new adapter
            decoder.lora_model(rank, alpha, dropout)
            logging.debug('created new adapter')

    optim = AdamW(decoder.parameters(), lr=lr)
    logging.debug('DECODER SIZE {}'.format(learnable_parameters(decoder.model)))
    logging.debug('MAPPER SIZE {}'.format(learnable_parameters(decoder.mapper)))

    training_losses = []
    validation_losses = []

    if log_step is None:
        log_step = len(train_loader)

    # training loop
    for epoch in range(epochs):
        log_loss = []
        i = 0
        # print(f'batches {len(train_loader)}')
        for batch in tqdm(train_loader, total=len(train_loader), desc='Epoch {}'.format(epoch)):
            i += 1
            optim.zero_grad()
            embeddings = encoder(batch['image'], trainable=frozen_encoder)
            output = decoder(embeddings, batch['caption'])
            loss = output.loss
            loss.backward()
            optim.step()
            loss = loss.detach().cpu().item()
            log_loss.append(loss)

            # logging and validation
            if (i + 1) % log_step == 0 or i == len(train_loader)-1:
                logging.debug('Logging step {}'.format(i + 1))
                # validation
                log_val_losses = []
                with torch.no_grad():
                    # noise may be used during training
                    decoder.add_noise = False
                    for val_batch in val_loader:
                        # validate using text embeddings in text only training
                        flag = True if dataset == 'petro-txt' else False
                        logging.debug(f'validation using text embedding? {flag}')

                        with torch.no_grad():
                            embeddings = encoder(val_batch['image'])
                            val_output = decoder(embeddings, val_batch['caption'])
                            log_val_losses.append(val_output.loss.detach().cpu().item())

                # save step loss and clean list
                validation_losses.append(sum(log_val_losses) / len(log_val_losses))
                training_losses.append(sum(log_loss) / len(log_loss))
                log_loss = []

                # plot and save loss history
                plt.plot(range(len(training_losses)), training_losses, label='training')
                plt.plot(range(len(validation_losses)), validation_losses, label='validation')
                plt.legend()
                plt.xlabel('step')
                plt.ylabel('loss')
                plt.title(f'training loss')

                plt.savefig(f'{root}/loss_plot.png')

                plt.clf()
                log = {'training_loss': training_losses, 'validation_loss': validation_losses}
                with open(f'{root}/loss_log.pkl', 'w') as f:
                    json.dump(log, f)

                decoder.train(True)
                decoder.add_noise = add_noise
                logging.debug(f'add noise to embeddings? {decoder.add_noise}')
                # model_size(decoder)
                # learnable_parameters(decoder)

        # epoch model
        model_dict = {'epoch': epoch + 1,
                      'model_state_dict': decoder.state_dict(),
                      'optimizer_state_dict': optim.state_dict(),
                      'loss': training_losses[-1]
                      }
        if save_history:
            torch.save(model_dict, f'{root}/checkpoint_{epoch+1}.pt')
        else:
            torch.save(model_dict, f'{root}/checkpoint.pt')


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
    parser.add_argument('--frozen_encoder', action='store_true', help='frozen encoder', default=False)

    args = parser.parse_args()
    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        logging.info(f'folders created: {args.save_dir}')

    precision = torch.float16 if args.dtype == 'fp16' else torch.float32
    logging.debug(f'precision: {precision}')
    cfg_path = os.path.join(args.decoder_name, 'adapter_config.json')

    if os.path.exists(cfg_path):
        logging.debug('decoder was adapted before locally')
        with open(cfg_path, 'rb') as f:
            cfg = json.load(f)
            args.rank = cfg['r']
            args.alpha = cfg['lora_alpha']
            logging.debug(f'decoder rank: {args.rank} alpha: {args.alpha}')

    train(args)

    result_dict = args.__dict__
    result_dict['checkpoint_path'] = os.path.join(args.save_path, 'checkpoint.pt')
    with open(f'{args.save_path}/experiment.json', 'w') as f:
        json.dump(result_dict, f, indent=2)
        logging.info(f'experiment saved')
