import time
import torch
import wandb
import nltk
import os

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from torchvision import transforms

from dataset import MemeDatasetFromFile
from logger import Logger
from models_captioning import EncoderCNN, DecoderLSTM
from utils_functions import save_checkpoint, load_checkpoint, load_config
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.nn.utils.rnn import pack_padded_sequence

torch.manual_seed(0)

# os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128"  # Proxy to train with hyperion

print(torch.cuda.is_available())
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
    # visualize Bahdanau attention
    image = transforms.ToPILImage()(image[0].unsqueeze(0).squeeze(0))
    image = image.resize([16 * 24, 16 * 24], Image.LANCZOS)
    # Only plot first element from batch
    caption = seq[0]
    words = caption
    for t in range(len(words)):
        if t > 50:
            break
        fig = plt.figure()
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[0, t, :]
        current_alpha = current_alpha.view(-1, 16, 16).squeeze(0)
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.cpu().numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.cpu().numpy(), [16 * 24, 16 * 24])

        plt.imshow(alpha, alpha=0.65, cmap='Greys_r')
        plt.axis('off')
        if mode == "cap":
            wandb.log({"Caption Attention": wandb.Image(fig)})
        elif mode == "img":
            wandb.log({"Image Caption Attention": wandb.Image(fig)})
        plt.close(fig)


def visualize_encoder_attention(img, attention_map):
    """
    Visualize attention map on the image
    img: [3, W, H] PyTorch tensor (image)
    attention_map: [W*H, W*H] PyTorch tensor (attention map)
    """
    img = img.squeeze(0) # check shape
    attention_map = attention_map.squeeze(0)
    img = img.permute(1, 2, 0)  # [N, N, 3]
    attention_map = attention_map.cpu().detach().numpy()
    attention_map = np.mean(attention_map, axis=0)  # Average over first dimension

    attention_map = attention_map.reshape(int(np.sqrt(attention_map.shape[0])),
                                          int(np.sqrt(attention_map.shape[0])))  # Reshape to W * H
    attention_map = (attention_map - np.min(attention_map)) / (
                np.max(attention_map) - np.min(attention_map))  # Normalisation

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img.cpu().detach().numpy())
    plt.title('Image')

    plt.subplot(1, 2, 2)
    im = plt.imshow(attention_map, cmap='Greys_r')  # Add attention map
    plt.title('Attention Map')
    plt.colorbar(im, fraction=0.046, pad=0.04)  # Add colorbar as legend
    # plt.close(fig)
    wandb.log({"Encoder Attention": wandb.Image(fig)})

def evaluate(encoder, decoder_cap, input_tensor, caption, voc, mode="val", length=80, plot_encoder_attention=False):
    with torch.no_grad():
        if mode == "train":
            max_cap = length
            target_cap = caption
            plot_feature = False
        else:
            max_cap = length
            target_cap = None
            plot_feature = True

        encoder_outputs,weights = encoder(input_tensor, plot_feature)
        if plot_encoder_attention:
            visualize_encoder_attention(input_tensor[0], weights[0])

        caption_output, _, attention_weights = decoder_cap(encoder_outputs, caption,
                                                           target_tensor=target_cap, max_caption=max_cap)
        caption_output = F.log_softmax(caption_output, dim=-1)
        decoded_caption = token2text(caption_output, voc)
    return decoded_caption, attention_weights


def calculate_bleu(target, predicted):
    references = [[caption] for caption in target]
    hypotheses = [caption for caption in predicted]
    # Calculate BLEU score
    bleu_score = nltk.translate.bleu_score.corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                                                       smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)
    return bleu_score


def token2text(output, voc):
    EOS_token = 2
    _, topi = output.topk(1)
    decoded_ids = topi.squeeze(-1)
    decoded_words_total = []
    limit = decoded_ids.size(0)
    for i in range(limit):
        decoded_words = []
        caption = decoded_ids[i]

        for idx in caption:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(voc.index2word[idx.item()])
        decoded_words_total.append(decoded_words)
    return decoded_words_total


def target2text(target, voc):  # for BLEU EVALUATION
    EOS_token = 2
    decoded_words_total = []
    limit = target.size(0)
    for i in range(limit):
        decoded_words = []
        caption = target[i]

        for idx in caption:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(voc.index2word[idx.item()])
        decoded_words_total.append(decoded_words)
    return decoded_words_total


def val_epoch(dataloader, encoder, decoder_cap, decoder_img, criterion, output_lang,  plot_encoder_attention=False,
              plot_decoder_attention=False):
    total_loss = 0
    total_samples = 0
    bleu_total = 0
    bleu_total_img = 0

    for data in dataloader:
        images = data["image"]
        meme_captions = data["meme_captions"].squeeze(1)
        img_captions = data["img_captions"].squeeze(1)
        max_caption = data["max_caption"]
        max_img = data["max_img"]
        total_samples += 1
        with torch.no_grad():
            encoder_outputs, _ = encoder(images, False)
            caption_outputs, _, attention_weights = decoder_cap(encoder_outputs, meme_captions,
                                                                None, max_caption)
            img_outputs, _, attention_weights_img = decoder_img(encoder_outputs, img_captions,
                                                                None, max_img)
        scores = pack_padded_sequence(caption_outputs, max_caption, batch_first=True, enforce_sorted=False)[0]
        targets = pack_padded_sequence(meme_captions, max_caption, batch_first=True, enforce_sorted=False)[0]
        loss = criterion(scores, targets)

        hypothesis = target2text(meme_captions, output_lang)
        references = F.log_softmax(caption_outputs, dim=-1)
        references = token2text(references, output_lang)
        bleu_score = calculate_bleu(hypothesis, references)
        bleu_loss = bleu_score

        scores = pack_padded_sequence(img_outputs, max_img, batch_first=True, enforce_sorted=False)[0]
        targets = pack_padded_sequence(img_captions, max_img, batch_first=True, enforce_sorted=False)[0]
        loss_img = criterion(scores, targets)

        hypothesis = target2text(img_captions, output_lang)
        references = F.log_softmax(img_outputs, dim=-1)
        references = token2text(references, output_lang)
        bleu_score_img = calculate_bleu(hypothesis, references)
        bleu_loss_img = bleu_score_img

        loss_attention = 0

        l2_norm = sum(torch.norm(p, 2) ** 2 for p in decoder_cap.parameters())
        loss_mix = loss + loss_img + loss_attention

        if total_samples % 20 == 0:
            captions, attention_weights = evaluate(encoder, decoder_cap, images, meme_captions,
                                                   output_lang, mode="val", length=max_caption,plot_encoder_attention=plot_encoder_attention)
            img_caption, attention_weights_img = evaluate(encoder, decoder_img, images, img_captions,
                                                          output_lang, mode="val", length=max_img, plot_encoder_attention=plot_encoder_attention)
            if plot_decoder_attention:
                visualize_att(images, captions, attention_weights, mode="cap")

            total_text = ""
            for caption in captions:
                converted_list = map(str, caption)
                result = ' '.join(converted_list)
                result += "/"
                total_text += result
            print(f"DataPoint Prediction: {total_samples}, Text: {total_text}")

            total_target = ""
            targets_dec = target2text(meme_captions, output_lang)
            for caption in targets_dec:
                converted_list = map(str, caption)
                result = ' '.join(converted_list)
                result += "/"
                total_target += result
            print(f"Datapoint Target: {total_target}")

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
            print(f"DataPoint IMG Prediction: {total_samples}, Text: {total_text}")

            total_target = ""
            targets_dec = target2text(img_captions, output_lang)
            for caption in targets_dec:
                converted_list = map(str, caption)
                result = ' '.join(converted_list)
                result += "/"
                total_target += result
            print(f"Datapoint IMG Target: {total_target}")

            columns = ["Datapoint", "Output"]
            data = [[str(total_samples), total_text]]
            table = wandb.Table(data=data, columns=columns)
            wandb.log({"Validation example Img": table})

        total_loss += loss_mix.item()
        bleu_total += bleu_loss
        bleu_total_img += bleu_loss_img

    return total_loss / len(dataloader), bleu_total / len(dataloader), bleu_total_img / len(dataloader)


def train_epoch(dataloader, encoder, decoder_cap, decoder_img, encoder_optimizer,
                decoder_optimizer_cap, decoder_optimizer_img, criterion, output_lang):
    total_loss = 0
    total_samples = 0
    for data in dataloader:
        images = data["image"]
        meme_captions = data["meme_captions"].squeeze(1)
        img_captions = data["img_captions"].squeeze(1)
        max_caption = data["max_caption"]
        max_img = data["max_img"]

        encoder_outputs,_ = encoder(images)
        caption_outputs, _, attention_weights = decoder_cap(encoder_outputs, meme_captions,
                                                            meme_captions, max_caption)
        img_outputs, _, attention_weights_img = decoder_img(encoder_outputs, img_captions,
                                                            img_captions, max_img)
        scores = pack_padded_sequence(caption_outputs, max_caption, batch_first=True, enforce_sorted=False)[0]
        targets = pack_padded_sequence(meme_captions, max_caption, batch_first=True, enforce_sorted=False)[0]
        # output = F.log_softmax(scores, dim=-1)
        # _, topi = output.topk(1)
        # decoded_ids = topi.squeeze(-1)
        # print(decoded_ids)
        loss = criterion(scores, targets)

        scores = pack_padded_sequence(img_outputs, max_img, batch_first=True, enforce_sorted=False)[0]
        targets = pack_padded_sequence(img_captions, max_img, batch_first=True, enforce_sorted=False)[0]
        loss_img = criterion(scores, targets)

        loss_attention = 0

        loss_cap = loss
        loss_cap_img = loss_img
        loss_mix = loss_cap + loss_cap_img

        if total_samples % 100 == 0:
            print(f"Datapoint: {total_samples}, loss_mix: {loss_mix.item()}, loss_attention: {loss_attention}")

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
        grad_clip = 5.
        if grad_clip is not None:
            clip_gradient(decoder_optimizer_cap, grad_clip)
            clip_gradient(decoder_optimizer_img, grad_clip)
            clip_gradient(encoder_optimizer, grad_clip)
        decoder_optimizer_cap.step()
        decoder_optimizer_img.step()
        encoder_optimizer.step()

        total_loss += loss_mix.item()
        total_samples += 1

    return total_loss / len(dataloader)


def train(train_dataloader, val_dataloader, encoder, decoder_cap, decoder_img, n_epochs, logger, output_lang,
        decoder_learning_rate=1e-4, encoder_learning_rate = 1e-5,
        print_every=100, plot_every=100, plot_encoder_attention=False, plot_decoder_attention=False):

    best_score = float('-inf')

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    val_plot_losses = []
    val_print_loss_total = 0  # Reset every print_every
    val_plot_loss_total = 0  # Reset every plot_every

    bleu_plot_losses = []
    bleu_print_loss_total = 0  # Reset every print_every
    bleu_plot_loss_total = 0  # Reset every plot_every

    bleu_plot_losses_img = []
    bleu_print_loss_total_img = 0  # Reset every print_every
    bleu_plot_loss_total_img = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(params=encoder.parameters(), lr=encoder_learning_rate)
    decoder_optimizer_cap = optim.Adam(params= decoder_cap.parameters(),
                                       lr=decoder_learning_rate)
    decoder_optimizer_img = optim.Adam(params=decoder_img.parameters(),
                                       lr=decoder_learning_rate)

    criterion = nn.CrossEntropyLoss(ignore_index=3).to(device) # no padding present, ignore UNK Token

    for epoch in range(1, n_epochs + 1):
        print(epoch)
        encoder.train()
        decoder_cap.train()
        decoder_img.train()
        loss = train_epoch(train_dataloader, encoder, decoder_cap, decoder_img,
                           encoder_optimizer, decoder_optimizer_cap, decoder_optimizer_img,
                           criterion,
                           output_lang)

        encoder.eval()
        decoder_cap.eval()
        decoder_img.eval()
        val_loss, bleu_loss, bleu_loss_img = val_epoch(val_dataloader, encoder, decoder_cap,
        decoder_img, criterion,output_lang, plot_encoder_attention=plot_encoder_attention, plot_decoder_attention=plot_decoder_attention)

        print_loss_total += loss
        plot_loss_total += loss

        val_print_loss_total += val_loss
        val_plot_loss_total += val_loss

        bleu_print_loss_total += bleu_loss
        bleu_plot_loss_total += bleu_loss

        bleu_print_loss_total_img += bleu_loss_img
        bleu_plot_loss_total_img += bleu_loss_img

        # Save best validation model on BLEU Metric
        if bleu_loss_img > best_score:
            save_checkpoint(decoder_cap, "MEMES_test_decoder_Cap")
            save_checkpoint(decoder_img, "MEMES_test_decoder_img")
            save_checkpoint(encoder, "MEMES_test_encoder")
            best_score = bleu_loss_img

        if epoch == n_epochs:
            save_checkpoint(decoder_cap, "FINAL_test_Cap")
            save_checkpoint(decoder_img, "FINAL_test_decoder_img")
            save_checkpoint(encoder, "FINAL_test_encoder")

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f"Train Epoch: {epoch}/{n_epochs}, Loss {print_loss_avg}")

            val_print_loss_avg = val_print_loss_total / print_every
            val_print_loss_total = 0
            print(f"Validation Epoch: {epoch}/{n_epochs}, Loss {val_print_loss_avg}")

            bleu_print_loss_avg = bleu_print_loss_total / print_every
            bleu_print_loss_total = 0
            print(f"Validation BLEU Epoch: {epoch}/{n_epochs}, Loss {bleu_print_loss_avg}")

            bleu_print_loss_avg_img = bleu_print_loss_total_img / print_every
            bleu_print_loss_total_img = 0
            print(f"Validation BLEU IMG Epoch: {epoch}/{n_epochs}, Loss {bleu_print_loss_avg_img}")

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            logger.log({'train_loss_avg': plot_loss_avg})
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

            val_plot_loss_avg = val_plot_loss_total / plot_every
            logger.log({'val_loss_avg': val_plot_loss_avg})
            val_plot_losses.append(val_plot_loss_avg)
            val_plot_loss_total = 0

            bleu_plot_loss_avg = bleu_plot_loss_total / plot_every
            logger.log({'val_bleu_avg': bleu_plot_loss_avg})
            bleu_plot_losses.append(bleu_plot_loss_avg)
            bleu_plot_loss_total = 0

            bleu_plot_loss_avg_img = bleu_plot_loss_total_img / plot_every
            logger.log({'val_bleu_avg_img': bleu_plot_loss_avg_img})
            bleu_plot_losses_img.append(bleu_plot_loss_avg_img)
            bleu_plot_loss_total_img = 0


def main():
    config = load_config()
    model_setting = config['model']
    train_setting = config['train']

    path_train = "data/memes-trainval.json"

    train_dataset = MemeDatasetFromFile(path_train)
    train_len = int(len(train_dataset) * 0.9)
    train_set, val_set = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_setting['batch_size'], shuffle=True)
    print("Training")
    print(len(train_set))
    print("Validation")
    print(len(val_set))
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=train_setting['batch_size'], shuffle=True)

    encoder = EncoderCNN(backbone=model_setting['encoder_model_type'], attention=model_setting['encoder_attention']).to(device)
    decoder_cap = DecoderLSTM(hidden_size=512, embed_size=300, output_size=train_dataset.n_words, num_layers=1, attention=model_setting['decoder_bahdanau']).to(
        device)
    decoder_img = DecoderLSTM(hidden_size=512, embed_size=300, output_size=train_dataset.n_words, num_layers=1, attention=model_setting['decoder_bahdanau']).to(
        device)

    wandb_logger = Logger(f"MEMES_Final-efficientnetB5-BA-selfAttention",
                          project='INM706-TEST', model=decoder_cap)
    logger = wandb_logger.get_logger()

    print("\n############## MODEL SETTINGS ##############")
    print(model_setting)
    print()
    print("\n############## TRAIN SETTINGS ##############")
    print(train_setting)
    print()
    n_epochs = train_setting['epochs']
    print(f"Length of vocabulary: {train_dataset.n_words}")


    train(train_dataloader, val_dataloader, encoder, decoder_cap, decoder_img, n_epochs, logger, train_dataset,
          print_every=1,plot_every=1, encoder_learning_rate=train_setting['encoder_learning_rate'], decoder_learning_rate=train_setting['decoder_learning_rate'],
          plot_encoder_attention=model_setting['encoder_attention'], plot_decoder_attention=model_setting['decoder_bahdanau'])
    return


if __name__ == '__main__':
    main()
