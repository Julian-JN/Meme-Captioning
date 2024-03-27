import time
import torch
import wandb
import os

import torch.nn as nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import skimage.transform
import matplotlib.cm as cm
from torchvision import transforms

from dataset import MemeDatasetFromFile
from logger import Logger
from models_captioning import EncoderCNN, DecoderLSTM
from utils_functions import save_checkpoint, load_checkpoint
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

torch.manual_seed(0)

# os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128"  # Proxy to train with hyperion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device set to {0}'.format(device))


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def visualize_att(image, seq, alphas, smooth=False, mode="cap"):
    image = transforms.ToPILImage()(image.squeeze(0))
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
    caption = seq[0]
    print(caption)
    words = caption

    for t in range(len(words)):
        if t > 50:
            break
        # plt.subplot(int(np.ceil(len(words) / 5.)), 5, t + 1)
        fig = plt.figure()
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        # print(alphas[t, :].shape)
        # current_alpha = alphas[t, :].view(-1, 14, 14).squeeze(0).mean(0)
        current_alpha = alphas[t, :].view(-1, 14, 14).squeeze(0)

        # print(current_alpha.shape)
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.cpu().numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.cpu().numpy(), [14 * 24, 14 * 24])
        if t == 0:
            # print(alpha.shape)
            plt.imshow(alpha, alpha=0.65, cmap='Greys_r')
        else:
            plt.imshow(alpha, alpha=0.65, cmap='Greys_r')
        plt.axis('off')
        if mode == "cap":
            wandb.log({"Caption Attention": wandb.Image(fig)})
        elif mode == "img":
            wandb.log({"Image Caption Attention": wandb.Image(fig)})
        # plt.show()
        plt.close(fig)

def evaluate(encoder, decoder_cap, decoder_img, input_tensor, caption, img_caption, voc, mode="val"):
    with torch.no_grad():
        if mode == "train":
            max_cap = caption.size(2)
            max_img = img_caption.size(2)
            target_cap = caption
            target_img = img_caption
            plot_feature = True
        else:
            max_cap = 80
            max_img = 80
            target_cap = None
            target_img = None
            plot_feature = False

        encoder_outputs = encoder(input_tensor, plot_feature)
        caption_output, _, attention_weights = decoder_cap(encoder_outputs, caption,
                                                           target_tensor=target_cap, max_caption=max_cap)
        img_output, _, attention_weights_img = decoder_img(encoder_outputs, img_caption,
                                                           target_tensor=target_img, max_caption=max_img)
        decoded_caption = token2text(caption_output, voc)
        decoded_img = token2text(img_output, voc)
    return decoded_caption, decoded_img, attention_weights, attention_weights_img


def token2text(output, voc):
    EOS_token = 2

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


def val_epoch(dataloader, encoder, decoder_cap, decoder_img, criterion, output_lang):
    total_loss = 0
    batch_loss = 0.0
    total_samples = 0
    for data in dataloader:
        images = data["image"]
        if not torch.is_tensor(images) or data["valid"] == False:
            continue
        titles = data["title"]
        meme_captions = data["meme_captions"]
        max_caption = data["max_caption"]
        max_img = data["max_img"]
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
            caption_outputs, _, attention_weights = decoder_cap(encoder_outputs, meme_captions,
                                                                      None, max_caption)
            img_outputs, _, attention_weights_img = decoder_img(encoder_outputs, img_captions,
                                                                None, max_img)
        loss = criterion(
            caption_outputs.view(-1, caption_outputs.size(-1)),
            meme_captions[:, :].view(-1)
        )

        loss_img = criterion(
            img_outputs.view(-1, img_outputs.size(-1)),
            img_captions[:, :].view(-1)
        )

        attention_mean = attention_weights.size(1)
        # print(attention_mean)
        # print(attention_weights.shape)
        loss_attention_cap = 1. * (((1. - attention_weights.sum(0)) ** 2).sum(0) / attention_mean)
        loss_attention_img = 1. * (((1. - attention_weights_img.sum(0)) ** 2).sum(0) / attention_mean)
        loss_attention = loss_attention_cap + loss_attention_img

        # l2_norm = sum(torch.norm(p, 2) ** 2 for p in decoder.parameters())
        loss_mix = loss + loss_img

        if total_samples % 500 == 0:
            captions, img_caption, attention_weights, attention_weights_img = evaluate(encoder, decoder_cap,decoder_img, images, meme_captions, img_captions,
                                                                output_lang, mode="val")
            total_text = ""
            for caption in captions:
                converted_list = map(str, caption)
                result = ' '.join(converted_list)
                result += "/"
                total_text += result
            print(f"DataPoint: {total_samples}, Text: {total_text}")

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
            print(f"DataPoint IMG: {total_samples}, Text: {total_text}")

            columns = ["Datapoint", "Output"]
            data = [[str(total_samples), total_text]]
            table = wandb.Table(data=data, columns=columns)
            wandb.log({"Validation example Img": table})

        total_loss += loss_mix.item()
        batch_loss += loss_mix.item()

    return total_loss / len(dataloader)


def train_epoch(dataloader, encoder, decoder_cap, decoder_img, encoder_optimizer,
                decoder_optimizer_cap, decoder_optimizer_img, criterion, output_lang, logger):
    total_loss = 0
    batch_loss = 0.0
    total_samples = 0
    for data in dataloader:
        images = data["image"]
        if not torch.is_tensor(images) or data["valid"] == False:
            continue
        titles = data["title"]
        meme_captions = data["meme_captions"]
        img_captions = data["img_captions"]
        max_caption = data["max_caption"]
        max_img = data["max_img"]

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

        encoder_outputs = encoder(images)
        caption_outputs, _, attention_weights = decoder_cap(encoder_outputs, meme_captions,
                                                            meme_captions, max_caption)
        img_outputs, _, attention_weights_img = decoder_img(encoder_outputs, img_captions,
                                                            img_captions, max_img)
        # print(caption_outputs.view(-1, caption_outputs.size(-1)).shape)
        # print(meme_captions[:, :].view(-1).shape)
        # print(meme_captions[:, :].view(-1))
        loss = criterion(
            caption_outputs.view(-1, caption_outputs.size(-1)),
            meme_captions[:, :].view(-1)
        )

        loss_img = criterion(
            img_outputs.view(-1, img_outputs.size(-1)),
            img_captions[:, :].view(-1)
        )

        attention_mean = attention_weights.size(1)
        # print(attention_mean)
        # print(attention_weights.shape)
        loss_attention_cap = 1. * (((1. - attention_weights.sum(0)) ** 2).sum(0) / attention_mean)
        loss_attention_img = 1. * (((1. - attention_weights_img.sum(0)) ** 2).sum(0) / attention_mean)
        loss_attention = loss_attention_cap + loss_attention_img

        # if loss_attention > 1:
        #     print("ABOVE!")
        #     for t in range(max_caption):
        #         print(t)
        #         print(attention_weights[t])
        #         print(attention_weights[t].sum())
        #     print("Attention")
        #     print(max_caption)
        #     print(attention_weights.sum(0).shape)
        #     print(attention_weights.sum(0).sum(0))
        #     print(attention_weights.sum(0).sum(0)/196)
        #     print(((1. - attention_weights.sum(0))**2))
        #     print(loss_attention)
        #     print(torch.mean(((1. - attention_weights.sum(0))**2)))

        # l2_norm = sum(torch.norm(p, 2) ** 2 for p in decoder.parameters())
        # loss_mix = loss + 0.8*loss_img + 0.0001*l2_norm
        loss_cap = loss
        loss_cap_img = loss_img
        loss_mix = loss_cap + loss_cap_img

        if total_samples % 100 == 0:
            print(
                f"Datapoint: {total_samples}, loss_mix {loss_mix.item()}, loss_attention {loss_attention}")

        if torch.isnan(loss):
            print("NAN DETECTED")
            print(caption_outputs)
            print(img_outputs)
            nan_mask = torch.isnan(caption_outputs)
            print(nan_mask)

        decoder_optimizer_cap.zero_grad()
        decoder_optimizer_img.zero_grad()
        encoder_optimizer.zero_grad()
        loss_cap.backward(retain_graph=True)
        loss_cap_img.backward(retain_graph=True)
        loss_mix.backward()
        grad_clip = 1.
        if grad_clip is not None:
            clip_gradient(decoder_optimizer_cap, grad_clip)
            clip_gradient(decoder_optimizer_img, grad_clip)
            clip_gradient(encoder_optimizer, grad_clip)
        decoder_optimizer_cap.step()
        decoder_optimizer_img.step()
        encoder_optimizer.step()

        total_loss += loss_mix.item()
        batch_loss += loss_mix.item()

        total_samples += 1
        if total_samples % 1000 == 0:
            total_text = ""
            captions, img_caption, attention_weights, attention_weights_img = evaluate(encoder, decoder_cap,
                                                                                       decoder_img, images,
                                                                                       meme_captions, img_captions,
                                                                                       output_lang, mode="train")
            visualize_att(images, captions, attention_weights, mode="cap")
            visualize_att(images, img_caption, attention_weights_img, mode="img")

            for caption in captions:
                converted_list = map(str, caption)
                result = ' '.join(converted_list)
                result += "/"
                total_text += result
            print(f"Datapoint: {total_samples}, Text: {total_text}")

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
            print(f"DataPoint IMG: {total_samples}, Text: {total_text}")

            columns = ["Datapoint", "Output"]
            data = [[str(total_samples), total_text]]
            table = wandb.Table(data=data, columns=columns)
            wandb.log({"Train example Img": table})

    return total_loss / len(dataloader)


def train(
        train_dataloader, val_dataloader, encoder, decoder_cap, decoder_img, n_epochs, logger, output_lang,
        learning_rate=0.0001,
        print_every=100, plot_every=100):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    val_plot_losses = []
    val_print_loss_total = 0  # Reset every print_every
    val_plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.00001)
    decoder_optimizer_cap = optim.Adam(decoder_cap.parameters(), lr=learning_rate)
    decoder_optimizer_img = optim.Adam(decoder_img.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=0)

    for epoch in range(1, n_epochs + 1):
        print(epoch)
        loss = train_epoch(train_dataloader, encoder.train(), decoder_cap.train(), decoder_img.train(),
                           encoder_optimizer, decoder_optimizer_cap, decoder_optimizer_img,
                           criterion,
                           output_lang, logger)

        val_loss = val_epoch(val_dataloader, encoder.eval(), decoder_cap.eval(), decoder_img.eval(), criterion,
                             output_lang)

        print_loss_total += loss
        plot_loss_total += loss

        val_print_loss_total += val_loss
        val_plot_loss_total += val_loss

        if epoch % print_every == 0:
            if epoch % 5 == 0:
                save_checkpoint(epoch, decoder_cap, "LSTM_Captions_decoder_Cap", decoder_optimizer_cap)
                save_checkpoint(epoch, decoder_img, "LSTM_Captions_decoder_img", decoder_optimizer_img)
                save_checkpoint(epoch, encoder, "LSTM_Captions_encoder", encoder_optimizer)
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
    hidden_size = 128
    batch_size = 1
    n_epochs = 30

    path_test = "data/memes-test.json"
    path_train = "data/memes-trainval.json"

    train_dataset = MemeDatasetFromFile(path_train)
    train_len = int(len(train_dataset) * 0.9)
    train_set, val_set = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
    print("Training")
    print(len(train_set))
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False)

    encoder = EncoderCNN(backbone='resnet').to(device)
    decoder_cap = DecoderLSTM(hidden_size=512, output_size=train_dataset.n_words, num_layers=1).to(device)
    decoder_img = DecoderLSTM(hidden_size=512, output_size=train_dataset.n_words, num_layers=1).to(device)

    wandb_logger = Logger(f"inm706_coursework_cnn_lstm_no_teach_resnet101_att_vis_duel_rcnn",
                          project='inm706_cw_hyperion_no_teach_resnet101_att_vis_duel_rcnn', model=decoder_img)
    logger = wandb_logger.get_logger()

    train(train_dataloader, val_dataloader, encoder, decoder_cap, decoder_img, n_epochs, logger, train_dataset,
          print_every=1,
          plot_every=1)
    return


if __name__ == '__main__':
    main()
