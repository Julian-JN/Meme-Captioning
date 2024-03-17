import time
import torch
import wandb
import os

import torch.nn as nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dataset import MemeDatasetFromFile
from logger import Logger
from models import EncoderCNN, DecoderLSTM
from utils import save_checkpoint, load_checkpoint

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

torch.manual_seed(0)

# os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128"  # Proxy to train with hyperion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device set to {0}'.format(device))


def evaluate(encoder, decoder, input_tensor, caption, img_caption, voc):
    with torch.no_grad():
        encoder_outputs = encoder(input_tensor)
        caption_output, img_output, _ = decoder(encoder_outputs, caption, img_caption, caption, img_caption)
        decoded_caption = token2text(caption_output, voc)
        decoded_img = token2text(img_output, voc)
    return decoded_caption, decoded_img


def token2text(output, voc):
    EOS_token = 1

    _, topi = output.topk(1)
    decoded_ids = topi.squeeze()
    decoded_words_total = []
    if decoded_ids.ndim != 1:
        limit = decoded_ids.size(0)
    else:
        limit = 1
    for i in range(limit):
        decoded_words = []
        if decoded_ids.ndim != 1:
            caption = decoded_ids[i]
        else:
            caption = decoded_ids

        for idx in caption:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(voc.index2word[idx.item()])
        decoded_words_total.append(decoded_words)
    return decoded_words_total


def val_epoch(dataloader, encoder, decoder, criterion, output_lang):
    total_loss = 0
    batch_loss = 0.0
    total_samples = 0
    for data in dataloader:
        images = data["image"]
        # print(images.shape)
        if not torch.is_tensor(images) or data["valid"] == False:
            continue
        titles = data["title"]
        meme_captions = data["meme_captions"]

        total_samples += 1
        # print(total_samples)
        # caption = meme_captions[:, 0]
        # items = [item.item() for item in caption[0]]
        # EOS_token = 1
        # decoded_words = []
        # for idx in items:
        #     if idx == EOS_token:
        #         decoded_words.append('<EOS>')
        #         break
        #     decoded_words.append(output_lang.index2word[idx])
        # print("Target")
        # print(decoded_words)

        img_captions = data["img_captions"]
        with torch.no_grad():
            encoder_outputs = encoder(images)
            caption_outputs, img_outputs, _ = decoder(encoder_outputs, meme_captions, img_captions, meme_captions,
                                                      img_captions)

        loss = criterion(
            caption_outputs.view(-1, caption_outputs.size(-1)),
            meme_captions[:, :].view(-1)
        )

        loss_img = criterion(
            img_outputs.view(-1, img_outputs.size(-1)),
            img_captions[:, :].view(-1)
        )

        loss_mix = loss + loss_img

        if total_samples % 500 == 0:
            captions, img_caption = evaluate(encoder, decoder, images, meme_captions, img_captions, output_lang)
            total_text = ""
            for caption in captions:
                converted_list = map(str, caption)
                result = ' '.join(converted_list)
                result += "/"
                total_text += result
            print(total_text)

            columns = ["Datapoint", "Output"]
            data = [[str(total_samples), total_text]]
            table = wandb.Table(data=data, columns=columns)
            wandb.log({"Validation example Caption": table})

            total_text = ""
            for caption in img_caption:
                converted_list = map(str, caption)
                result = ' '.join(converted_list)
                result += "/"
                total_text += result
            print(total_text)

            columns = ["Datapoint", "Output"]
            data = [[str(total_samples), total_text]]
            table = wandb.Table(data=data, columns=columns)
            wandb.log({"Validation example Img": table})

        total_loss += loss_mix.item()
        batch_loss += loss_mix.item()

    return total_loss / len(dataloader)


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion, output_lang, logger):
    total_loss = 0
    batch_loss = 0.0
    total_samples = 0
    # decoder_optimizer.zero_grad()
    for data in dataloader:
        decoder_optimizer.zero_grad()
        images = data["image"]
        # print(images.shape)
        if not torch.is_tensor(images) or data["valid"] == False:
            # print("UNAVAILABLE IMAGE")
            continue
        titles = data["title"]
        meme_captions = data["meme_captions"]

        # caption = meme_captions[:, 0]
        # items = [item.item() for item in caption[0]]
        # print(items)
        # EOS_token = 1
        # decoded_words = []
        # for idx in items:
        #     if idx == EOS_token:
        #         decoded_words.append('<EOS>')
        #         break
        #     decoded_words.append(output_lang.index2word[idx])
        # print(decoded_words)

        img_captions = data["img_captions"]
        # print(img_captions.shape)
        # print(meme_captions.shape)
        encoder_outputs = encoder(images)
        caption_outputs, img_outputs, _ = decoder(encoder_outputs, meme_captions, img_captions, meme_captions,
                                        img_captions)


        loss = criterion(
            caption_outputs.view(-1, caption_outputs.size(-1)),
            meme_captions[:, :].view(-1)
        )

        # print(loss)

        loss_img = criterion(
            img_outputs.view(-1, img_outputs.size(-1)),
            img_captions[:, :].view(-1)
        )

        # print(loss_img)

        loss_mix = loss + loss_img

        print(loss_mix)

        if torch.isnan(loss):
            print("NAN DETECTED")
            print(caption_outputs)
            print(img_outputs)
            nan_mask = torch.isnan(caption_outputs)
            print(nan_mask)

        loss_mix.backward()
        decoder_optimizer.step()
        encoder_optimizer.step()

        total_loss += loss_mix.item()
        batch_loss += loss_mix.item()

        total_samples += 1
        if total_samples % 500 == 0:
            print(loss_mix)
            print(total_loss)
            total_text = ""
            captions, img_caption = evaluate(encoder, decoder, images, meme_captions, img_captions, output_lang)
            for caption in captions:
                converted_list = map(str, caption)
                result = ' '.join(converted_list)
                result += "/"
                total_text += result
            print(total_text)

            columns = ["Datapoint", "Output"]
            data = [[str(total_samples), total_text]]
            table = wandb.Table(data=data, columns=columns)
            wandb.log({"Train example Caption": table})

            total_text = ""
            for caption in img_caption:
                converted_list = map(str, caption)
                result = ' '.join(converted_list)
                result += "/"
                total_text += result
            print(total_text)

            columns = ["Datapoint", "Output"]
            data = [[str(total_samples), total_text]]
            table = wandb.Table(data=data, columns=columns)
            wandb.log({"Train example Img": table})

    return total_loss / len(dataloader)


def train(
        train_dataloader, val_dataloader, encoder, decoder, n_epochs, logger, output_lang, learning_rate=0.001,
        print_every=100, plot_every=100):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    val_plot_losses = []
    val_print_loss_total = 0  # Reset every print_every
    val_plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        print(epoch)
        loss = train_epoch(train_dataloader, encoder.train(), decoder.train(), encoder_optimizer, decoder_optimizer, criterion,
                           output_lang, logger)

        val_loss = val_epoch(val_dataloader, encoder.eval(), decoder.eval(), criterion,
                             output_lang)

        print_loss_total += loss
        plot_loss_total += loss

        val_print_loss_total += val_loss
        val_plot_loss_total += val_loss

        if epoch % print_every == 0:
            save_checkpoint(epoch, decoder, "LSTM_Captions", decoder_optimizer)
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f"Train Epoch: {epoch}/{n_epochs}, Loss {print_loss_avg}")

            val_print_loss_avg = val_print_loss_total / print_every
            val_print_loss_total = 0
            print(f"Validation Epoch: {epoch}/{n_epochs}, Loss {val_print_loss_avg}")

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            logger.log({'train_loss_avg': plot_loss_avg})
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

            val_plot_loss_avg = val_plot_loss_total / plot_every
            logger.log({'val_loss_avg': val_plot_loss_avg})
            val_plot_losses.append(val_plot_loss_avg)
            val_plot_loss_total = 0


def main():
    # dataset = TranslationDataset('eng', 'fra', True)
    hidden_size = 128
    batch_size = 1
    n_epochs = 30

    path_test = "data/memes-test.json"
    path_train = "data/memes-trainval.json"

    train_dataset = MemeDatasetFromFile(path_train)
    train_len = int(len(train_dataset) * 0.9)
    train_set, val_set = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)

    encoder = EncoderCNN(backbone='resnet').to(device)
    decoder = DecoderLSTM(hidden_size=512, output_size=train_dataset.n_words, num_layers=1).to(device)

    wandb_logger = Logger(f"inm706_coursework_cnn_lstm_experiment", project='inm706_cw_exp', model=decoder)
    logger = wandb_logger.get_logger()

    train(train_dataloader, val_dataloader, encoder, decoder, n_epochs, logger, train_dataset, print_every=1,
          plot_every=1)
    return


if __name__ == '__main__':
    main()
