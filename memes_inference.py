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

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


def visualize_att(image, seq, alphas, mode="cap"):
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
        if current_alpha.size(0) == 16*16:
            current_alpha = current_alpha.view(-1, 16, 16).squeeze(0)
        else:
            current_alpha = current_alpha.view(-1, 16, 16).squeeze(0).mean(0)  # mean if object detection included

        alpha = skimage.transform.resize(current_alpha.cpu().numpy(), [16 * 24, 16 * 24])

        plt.imshow(alpha, alpha=0.65, cmap='Greys_r')
        plt.axis('off')
        plt.show()
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
    plt.show()
    plt.close(fig)


def evaluate(encoder, decoder_cap, input_tensor, caption, voc, mode="val", length=80, plot_encoder_attention=False):
    with torch.no_grad():
        if mode == "train":
            max_cap = length
            target_cap = caption
            plot_feature = False
        else:
            max_cap = length
            target_cap = None
            plot_feature = False

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


def test_epoch(dataloader, encoder, decoder_cap, decoder_img, criterion, output_lang, plot_encoder_attention=False,
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
        all_captions = data["all_captions"].squeeze(1)
        all_captions_img = data["all_img_captions"].squeeze(1)

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
        if all_captions.size(1) == 80:
            hypothesis_all = hypothesis
        else:
            hypothesis_all = alltarget2text(all_captions, output_lang)
        references = F.log_softmax(caption_outputs, dim=-1)
        references = token2text(references, output_lang)
        bleu_score = calculate_bleu(hypothesis_all, references)
        bleu_loss = bleu_score


        scores = pack_padded_sequence(img_outputs, max_img, batch_first=True, enforce_sorted=False)[0]
        targets = pack_padded_sequence(img_captions, max_img, batch_first=True, enforce_sorted=False)[0]
        loss_img = criterion(scores, targets)

        hypothesis_img = target2text(img_captions, output_lang)
        if all_captions_img.size(1) == 80:
            hypothesis_all = hypothesis
        else:
            hypothesis_all = alltarget2text(all_captions_img, output_lang)
        references_img = F.log_softmax(img_outputs, dim=-1)
        references_img = token2text(references_img, output_lang)
        bleu_score_img = calculate_bleu(hypothesis_all, references_img)
        bleu_loss_img = bleu_score_img

        loss_attention = 0

        l2_norm = sum(torch.norm(p, 2) ** 2 for p in decoder_cap.parameters())
        loss_mix = loss + loss_img + loss_attention

        if total_samples % 50 == 0:

            plot_image = True
            if plot_image:
                converted_list = map(str, references_img[0])
                pred = ' '.join(converted_list)
                converted_list = map(str, hypothesis_img[0])
                target = ' '.join(converted_list)
                result = "Predicted: " + pred + "/ Target:" + target
                img = images.squeeze(0)
                img = img.permute(1, 2, 0)  # [N, N, 3]
                plt.imshow(img.cpu().detach().numpy())
                plt.title(result)
                plt.show()

            captions, attention_weights = evaluate(encoder, decoder_cap, images, meme_captions,
                                                   output_lang, mode="val", length=max_caption,plot_encoder_attention=plot_encoder_attention)
            img_caption, attention_weights_img = evaluate(encoder, decoder_img, images, img_captions,
                                                          output_lang, mode="val", length=max_img, plot_encoder_attention=plot_encoder_attention)
            if plot_decoder_attention:
                visualize_att(images, img_caption, attention_weights_img, mode="cap")

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


        total_loss += loss_mix.item()
        bleu_total += bleu_loss
        bleu_total_img += bleu_loss_img

    return total_loss / len(dataloader), bleu_total / len(dataloader), bleu_total_img / len(dataloader)


def test(test_dataloader, encoder, decoder_cap, decoder_img, output_lang,
          plot_encoder_attention=False, plot_decoder_attention=False):


    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)


    encoder.eval()
    decoder_cap.eval()
    decoder_img.eval()
    test_loss, metric_loss, metric_loss_img = test_epoch(test_dataloader, encoder, decoder_cap, decoder_img, criterion,output_lang, plot_encoder_attention=plot_encoder_attention, plot_decoder_attention=plot_decoder_attention)
    print(f"Inference: CE-Loss {test_loss}")
    print(f"Inference: BLEU-Loss Caption {metric_loss}")
    print(f"Inference: BLEU-Loss Caption {metric_loss_img}")



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
    n_epochs = train_setting['epochs']


    path_test = "data/memes-test.json"
    test_dataset = MemeDatasetFromFile(path_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    print(f"Length of vocabulary: {test_dataset.n_words}")

    encoder = EncoderCNN(backbone=model_setting['encoder_model_type'], attention=model_setting['encoder_attention']).to(
        device)
    decoder_cap = DecoderLSTM(hidden_size=512, embed_size=300, output_size=test_dataset.n_words, num_layers=1,
                              attention=model_setting['decoder_bahdanau']).to(device)
    decoder_img = DecoderLSTM(hidden_size=512, embed_size=300, output_size=test_dataset.n_words, num_layers=1,
                              attention=model_setting['decoder_bahdanau']).to(device)

    load_checkpoint(encoder, "train_checkpoint/FINAL-MEMES-EfficientB5-BA-selfAttention-LSTM_Captions_encoder_ckpt.pth")
    load_checkpoint(decoder_cap, "train_checkpoint/FINAL-MEMES-EfficientB5-BA-selfAttention-LSTM_Captions_decoder_Cap_ckpt.pth")
    load_checkpoint(decoder_img, "train_checkpoint/FINAL-MEMES-EfficientB5-BA-selfAttention-LSTM_Captions_decoder_img_ckpt.pth")


    test(test_dataloader, encoder, decoder_cap, decoder_img, test_dataset,
          plot_encoder_attention=model_setting['encoder_attention'],
          plot_decoder_attention=model_setting['decoder_bahdanau'])
    return


if __name__ == '__main__':
    main()
