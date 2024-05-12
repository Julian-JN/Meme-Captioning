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

from dataset_flickr import FlickrDataset
from logger import Logger
from models_captioning import EncoderCNN, DecoderLSTM
from utils_functions import save_checkpoint, load_checkpoint, load_config
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


def visualize_att(image, seq, alphas, mode="cap"):
    image = transforms.ToPILImage()(image[0].unsqueeze(0).squeeze(0))
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
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
        if current_alpha.size(0) == 196:
            current_alpha = current_alpha.view(-1, 14, 14).squeeze(0)
        else:
            current_alpha = current_alpha.view(-1, 14, 14).squeeze(0).mean(0)  # mean if object detection included

        alpha = skimage.transform.resize(current_alpha.cpu().numpy(), [14 * 24, 14 * 24])

        plt.imshow(alpha, alpha=0.65, cmap='Greys_r')
        plt.axis('off')
        if mode == "cap":
            wandb.log({"Caption Attention": wandb.Image(fig)})
        elif mode == "img":
            wandb.log({"Image Caption Attention": wandb.Image(fig)})
        # plt.show()
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

        encoder_outputs, weights = encoder(input_tensor, plot_feature)
        if plot_encoder_attention:
            visualize_encoder_attention(input_tensor[0], weights[0])
        caption_output, _, attention_weights = decoder_cap(encoder_outputs, caption,target_tensor=target_cap, max_caption=max_cap)
        caption_output = F.log_softmax(caption_output, dim=-1)
        decoded_caption = token2text(caption_output, voc)
    return decoded_caption, attention_weights


def calculate_bleu(target, predicted):
    references = [target]
    # references = [[caption] for caption in target]
    hypotheses = [caption for caption in predicted]
    # Calculate BLEU score
    bleu_score = nltk.translate.bleu_score.corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                                                       smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)
    return bleu_score


def calculate_meteor(target, predicted):
    references = [[caption] for caption in target]

    hypotheses = [caption for caption in predicted]
    # Calculate METEOR score
    meteor_score = nltk.translate.meteor_score.meteor_score(references, hypotheses)
    return meteor_score


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


def alltarget2text(target, voc):  # for BLEU EVALUATION
    EOS_token = 2
    decoded_words_total = []
    limit = target.size(0)
    for i in range(limit):
        captions = target[i]
        for caption in captions:
            decoded_words = []
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


def test_epoch(dataloader, encoder, decoder_cap, criterion, output_lang, plot_encoder_attention=False,
              plot_decoder_attention=False):
    total_loss = 0
    total_samples = 0
    bleu_total = 0
    for data in dataloader:
        images = data["image"]
        meme_captions = data["meme_captions"].squeeze(1)
        all_captions = data["all_captions"].squeeze(1)
        max_caption = data["max_caption"]
        total_samples += 1
        with torch.no_grad():
            encoder_outputs, _ = encoder(images, False)
            caption_outputs, _, attention_weights = decoder_cap(encoder_outputs, meme_captions, None, max_caption)

        scores = pack_padded_sequence(caption_outputs, max_caption, batch_first=True, enforce_sorted=False)[0]
        targets = pack_padded_sequence(meme_captions, max_caption, batch_first=True, enforce_sorted=False)[0]
        loss = criterion(scores, targets)

        hypothesis = target2text(meme_captions, output_lang)
        hypothesis_all = alltarget2text(all_captions, output_lang)
        references = F.log_softmax(caption_outputs, dim=-1)
        references = token2text(references, output_lang)
        bleu_score = calculate_bleu(hypothesis_all, references)


        loss_attention = 0
        # loss_attention = 1.0 * ((1. - attention_weights.sum(dim=1)) ** 2).mean()

        loss_mix = loss + loss_attention
        bleu_loss = bleu_score

        if total_samples % 500 == 0:

            plot_image = False
            if plot_image:
                converted_list = map(str, references[0])
                pred = ' '.join(converted_list)
                converted_list = map(str, hypothesis[0])
                target = ' '.join(converted_list)
                result = "Predicted: " + pred + "/ Target:" + target
                img = images.squeeze(0)
                img = img.permute(1, 2, 0)  # [N, N, 3]
                plt.imshow(img.cpu().detach().numpy())
                plt.title(result)
                plt.show()

            captions, attention_weights = evaluate(encoder, decoder_cap, images, meme_captions, output_lang, mode="val",
                                                   length=max_caption, plot_encoder_attention=plot_encoder_attention)
            if plot_decoder_attention:
                visualize_att(images, captions, attention_weights, mode="cap")
            total_text = ""
            for caption in captions:
                converted_list = map(str, caption)
                result = ' '.join(converted_list)
                result += "/"
                total_text += result
            print(f"DataPoint: {total_samples}, Text: {total_text}")

            total_target = ""
            targets_dec = target2text(meme_captions, output_lang)
            for caption in targets_dec:
                converted_list = map(str, caption)
                result = ' '.join(converted_list)
                result += "/"
                total_target += result
            print(f"Datapoint Target: {total_target}")

        total_loss += loss_mix.item()
        bleu_total += bleu_loss

    return total_loss / len(dataloader), bleu_total / len(dataloader)


def test(test_dataloader, encoder, decoder_cap, output_lang,
          plot_encoder_attention=False, plot_decoder_attention=False):


    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)


    encoder.eval()
    decoder_cap.eval()
    test_loss, metric_loss = test_epoch(test_dataloader, encoder, decoder_cap, criterion,output_lang, plot_encoder_attention=plot_encoder_attention, plot_decoder_attention=plot_decoder_attention)
    print(f"Inference: CE-Loss {test_loss}")
    print(f"Inference: BLEU-Loss {metric_loss}")



def main():
    config = load_config()
    model_setting = config['model']
    train_setting = config['train']
    print("\n############## MODEL SETTINGS ##############")
    print(model_setting)
    print()
    print("\n############## TRAIN SETTINGS ##############")
    print(train_setting)
    print()

    train_dataset = FlickrDataset()
    train_len = int(len(train_dataset) * 0.8)
    train_set, val_set = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
    val_len = int(len(val_set) * 0.5)
    val_set, test_set = random_split(val_set, [val_len, len(val_set) - val_len])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_setting['batch_size'], shuffle=True)
    print("Training")
    print(len(train_set))
    print("Validation")
    print(len(val_set))
    print("Test")
    print(len(test_set))
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=train_setting['batch_size'], shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

    print(f"Length of vocabulary: {train_dataset.n_words}")

    encoder = EncoderCNN(backbone=model_setting['encoder_model_type'], attention=model_setting['encoder_attention']).to(
        device)
    decoder_cap = DecoderLSTM(hidden_size=512, embed_size=300, output_size=train_dataset.n_words, num_layers=1,
                              attention=model_setting['decoder_bahdanau']).to(device)

    load_checkpoint(encoder, "train_checkpoint/Resnet-LSTM_Captions_encoder_ckpt.pth")
    load_checkpoint(decoder_cap, "train_checkpoint/Resnet-LSTM_Captions_decoder_Cap_ckpt.pth")

    test(test_dataloader, encoder, decoder_cap, train_dataset,
          plot_encoder_attention=model_setting['encoder_attention'],
          plot_decoder_attention=model_setting['decoder_bahdanau'])
    return


if __name__ == '__main__':
    main()
